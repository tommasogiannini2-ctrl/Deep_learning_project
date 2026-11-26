"""Test unitari per la loss (src/losses.py) e i componenti comuni dei modelli (src/models/common.py).

Tutti i test usano dati sintetici o il dummy batch del contratto: non serve DATA/.
Esecuzione:  python -m unittest discover tests
"""

from __future__ import annotations

import unittest

import torch

from src.config import load_config
from src.contract import N_DEFAULT_FEATURES, make_dummy_batch, validate_predictions
from src.losses import PinballLoss, build_loss, pinball_loss
from src.models.common import (
    ContinuousTimeEncoding,
    MonotonicQuantileHead,
    build_step_features,
    step_feature_dim,
)


class TestPinballLoss(unittest.TestCase):
    """Correttezza numerica e gestione delle maschere della Pinball Loss."""

    @staticmethod
    def _single_value(pred: float, true: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Un solo campione/step/indice, con lo stesso valore previsto per i 3 quantili."""
        preds = torch.full((1, 1, 1, 3), pred)
        target = torch.full((1, 1, 1), true)
        return preds, target

    def test_asymmetry_matches_hand_computation(self) -> None:
        # Valore vero 0.5. Per ogni quantile tau:
        #   sottostima di 0.1 (pred 0.4) -> costo tau * 0.1
        #   sovrastima di 0.1 (pred 0.6) -> costo (1 - tau) * 0.1
        preds, target = self._single_value(0.4, 0.5)
        per_q = pinball_loss(preds, target, reduction="none")[0, 0, 0]
        torch.testing.assert_close(per_q, torch.tensor([0.01, 0.05, 0.09]))

        preds, target = self._single_value(0.6, 0.5)
        per_q = pinball_loss(preds, target, reduction="none")[0, 0, 0]
        torch.testing.assert_close(per_q, torch.tensor([0.09, 0.05, 0.01]))

        # La media sui 3 quantili vale (0.01 + 0.05 + 0.09) / 3 = 0.05 in entrambi i casi.
        torch.testing.assert_close(pinball_loss(preds, target), torch.tensor(0.05))

    def test_perfect_prediction_has_zero_loss(self) -> None:
        target = torch.rand(4, 3, 3)
        preds = target.unsqueeze(-1).expand(-1, -1, -1, 3).clone()
        self.assertEqual(pinball_loss(preds, target).item(), 0.0)

    def test_masked_targets_are_ignored(self) -> None:
        torch.manual_seed(0)
        preds = torch.randn(2, 3, 3, 3, requires_grad=True)
        target = torch.randn(2, 3, 3)
        mask = torch.zeros(2, 3, 3, dtype=torch.bool)
        mask[0, 1, 2] = True  # un target mancante

        reference = pinball_loss(preds, target, target_mask=mask)

        # Mettiamo un valore assurdo (e anche NaN) nella posizione mascherata:
        # la loss non deve cambiare e non deve diventare NaN.
        corrupted = target.clone()
        corrupted[0, 1, 2] = float("nan")
        loss = pinball_loss(preds, corrupted, target_mask=mask)
        torch.testing.assert_close(loss, reference)

        # Il gradiente sulle predizioni del target mascherato è esattamente 0.
        loss.backward()
        self.assertTrue(torch.isfinite(preds.grad).all())
        self.assertTrue(torch.all(preds.grad[0, 1, 2] == 0))
        self.assertTrue(torch.any(preds.grad[0, 0, 0] != 0))

    def test_mean_uses_only_valid_elements(self) -> None:
        preds, target = self._single_value(0.4, 0.5)
        # Due indici: il primo valido, il secondo mascherato con un errore enorme.
        preds = torch.cat([preds, torch.full((1, 1, 1, 3), 100.0)], dim=2)
        target = torch.cat([target, torch.zeros(1, 1, 1)], dim=2)
        mask = torch.tensor([[[False, True]]])
        # La media deve contare solo i 3 elementi validi -> 0.05, non 0.025.
        torch.testing.assert_close(pinball_loss(preds, target, target_mask=mask), torch.tensor(0.05))

    def test_all_masked_returns_zero_with_graph(self) -> None:
        preds = torch.randn(2, 3, 3, 3, requires_grad=True)
        target = torch.randn(2, 3, 3)
        mask = torch.ones(2, 3, 3, dtype=torch.bool)
        loss = pinball_loss(preds, target, target_mask=mask)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()  # non deve sollevare errori
        self.assertTrue(torch.all(preds.grad == 0))

    def test_shape_errors_are_explicit(self) -> None:
        with self.assertRaises(ValueError):
            pinball_loss(torch.zeros(2, 3, 3), torch.zeros(2, 3, 3))  # preds senza asse quantili
        with self.assertRaises(ValueError):
            pinball_loss(torch.zeros(2, 3, 3, 3), torch.zeros(2, 3))  # target con shape errata
        with self.assertRaises(ValueError):
            pinball_loss(torch.zeros(2, 3, 3, 2), torch.zeros(2, 3, 3))  # 2 quantili invece di 3
        with self.assertRaises(ValueError):
            PinballLoss(quantiles=(0.9, 0.5, 0.1))  # quantili non crescenti

    def test_module_and_factory_match_function(self) -> None:
        batch = make_dummy_batch(batch_size=4)
        preds = torch.randn(4, 3, 3, 3)
        expected = pinball_loss(preds, batch["target"], target_mask=batch["target_mask"])

        criterion = build_loss(load_config())
        self.assertIsInstance(criterion, PinballLoss)
        torch.testing.assert_close(criterion(preds, batch["target"], batch["target_mask"]), expected)

        with self.assertRaises(ValueError):
            build_loss({"training": {"loss": "loss_inesistente"}})


class TestMonotonicQuantileHead(unittest.TestCase):
    """Forma dell'output, assenza di quantile crossing e previsione residua."""

    def test_output_contract_and_no_crossing_on_extreme_inputs(self) -> None:
        torch.manual_seed(0)
        head = MonotonicQuantileHead(in_features=16, residual=True)
        # Pesi e input casuali e molto grandi per mettere alla prova la monotonicità.
        with torch.no_grad():
            for p in head.parameters():
                p.normal_(0.0, 5.0)
        x = torch.randn(8, 3, 16) * 50.0
        last_obs = torch.randn(8, 3)

        preds = head(x, last_obs)
        validate_predictions(preds, batch_size=8, require_monotonic_quantiles=True)
        self.assertEqual(tuple(preds.shape), (8, 3, 3, 3))

    def test_initial_residual_head_equals_persistence(self) -> None:
        head = MonotonicQuantileHead(in_features=16, residual=True, init_spread=0.5)
        x = torch.randn(4, 3, 16)
        last_obs = torch.randn(4, 3)
        preds = head(x, last_obs)

        # Pesi a zero all'inizio: mediana = last_obs per ogni step, intervalli +-0.5.
        expected_median = last_obs.unsqueeze(1).expand(-1, 3, -1)
        torch.testing.assert_close(preds[..., 1], expected_median)
        torch.testing.assert_close(preds[..., 1] - preds[..., 0], torch.full((4, 3, 3), 0.5))
        torch.testing.assert_close(preds[..., 2] - preds[..., 1], torch.full((4, 3, 3), 0.5))

    def test_direct_head_ignores_last_obs(self) -> None:
        head = MonotonicQuantileHead(in_features=16, residual=False)
        x = torch.randn(4, 3, 16)
        preds = head(x)  # last_obs non necessaria
        validate_predictions(preds, batch_size=4, require_monotonic_quantiles=True)
        torch.testing.assert_close(preds[..., 1], torch.zeros(4, 3, 3))

    def test_residual_head_requires_last_obs(self) -> None:
        head = MonotonicQuantileHead(in_features=16, residual=True)
        with self.assertRaises(ValueError):
            head(torch.randn(4, 3, 16))

    def test_gradients_reach_the_head(self) -> None:
        head = MonotonicQuantileHead(in_features=16)
        x = torch.randn(4, 3, 16)
        loss = pinball_loss(head(x, torch.randn(4, 3)), torch.randn(4, 3, 3))
        loss.backward()
        # Anche con i pesi inizializzati a zero il gradiente sui pesi è non nullo.
        self.assertTrue(head.proj.weight.grad.abs().sum() > 0)


class TestStepFeaturesAndTimeEncoding(unittest.TestCase):
    """Input mask-aware e codifica temporale su feature variabili e sequenze lunghe."""

    def test_step_features_default_and_ablated(self) -> None:
        # F = 21 (default) e F = 20 (ablation con una feature rimossa).
        for num_features in (N_DEFAULT_FEATURES, N_DEFAULT_FEATURES - 1):
            batch = make_dummy_batch(batch_size=4, num_features=num_features)
            feats = build_step_features(
                batch["history"], batch["history_mask"], batch["history_delta_t"], batch["history_padding_mask"]
            )
            B, T, F = batch["history"].shape
            self.assertEqual(tuple(feats.shape), (B, T, step_feature_dim(num_features)))
            self.assertTrue(torch.isfinite(feats).all())

            # I flag di presenza (seconda metà) sono 1 esattamente dove il valore è osservato.
            presence = feats[..., F : 2 * F]
            torch.testing.assert_close(presence, (~batch["history_mask"]).float())

            # Sul padding tutti i canali (valori, presenza, tempo) sono 0.
            pad = batch["history_padding_mask"]
            if pad.any():
                self.assertTrue(torch.all(feats[pad] == 0))

    def test_step_features_rezero_values_under_mask(self) -> None:
        values = torch.ones(1, 2, 2)
        mask = torch.tensor([[[True, False], [False, False]]])
        delta_t = torch.tensor([[-1.0, 0.0]])
        feats = build_step_features(values, mask, delta_t)
        self.assertEqual(feats[0, 0, 0].item(), 0.0)  # valore mascherato riportato a 0
        self.assertEqual(feats[0, 0, 2].item(), 0.0)  # flag di presenza = 0
        self.assertEqual(feats[0, 0, 3].item(), 1.0)  # flag di presenza = 1

    def test_long_sequences_160_days(self) -> None:
        batch = make_dummy_batch(
            batch_size=3, min_hist_len=150, max_hist_len=160, min_fut_len=150, max_fut_len=160
        )
        encoder = ContinuousTimeEncoding(d_model=16)
        h_enc = encoder(batch["history_delta_t"], batch["history_padding_mask"])
        f_enc = encoder(batch["future_delta_t"], batch["future_padding_mask"])
        self.assertEqual(h_enc.shape[-1], 16)
        self.assertTrue(torch.isfinite(h_enc).all() and torch.isfinite(f_enc).all())

        feats = build_step_features(
            batch["future"], batch["future_mask"], batch["future_delta_t"], batch["future_padding_mask"]
        )
        self.assertTrue(torch.isfinite(feats).all())

    def test_time_encoding_distinguishes_days_and_masks_padding(self) -> None:
        encoder = ContinuousTimeEncoding(d_model=16)
        days = torch.arange(-160, 161, dtype=torch.float32)  # da -160 a +160 giorni
        enc = encoder(days)
        self.assertEqual(tuple(enc.shape), (days.numel(), 16))

        # Giorni diversi -> codifiche diverse (distanza minima tra tutte le coppie > 0).
        distances = torch.cdist(enc, enc) + torch.eye(days.numel()) * 1e6
        self.assertGreater(distances.min().item(), 1e-3)

        # Le posizioni di padding vengono azzerate.
        dt = torch.tensor([[1.0, 2.0, 0.0]])
        pad = torch.tensor([[False, False, True]])
        self.assertTrue(torch.all(encoder(dt, pad)[0, 2] == 0))

        # La codifica non ha parametri addestrabili.
        self.assertEqual(sum(p.numel() for p in encoder.parameters()), 0)

    def test_time_encoding_rejects_odd_dimension(self) -> None:
        with self.assertRaises(ValueError):
            ContinuousTimeEncoding(d_model=15)


if __name__ == "__main__":
    unittest.main()
