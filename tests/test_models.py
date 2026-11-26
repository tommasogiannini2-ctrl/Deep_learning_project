"""Test unitari per le baseline (src/models/baselines.py) e la factory (src/models/__init__.py).

Tutti i test usano il dummy batch del contratto: non serve DATA/.
Esecuzione:  python -m unittest discover tests
"""

from __future__ import annotations

import copy
import unittest

import torch

from src.config import load_config
from src.contract import N_DEFAULT_FEATURES, make_dummy_batch, validate_predictions
from src.losses import PinballLoss
from src.models import MODEL_REGISTRY, PersistenceModel, ResidualMLPBaseline, build_model, infer_num_features
from src.models.common import gather_time_steps


def _make_mlp(num_features: int = N_DEFAULT_FEATURES, **kwargs) -> ResidualMLPBaseline:
    """MLP in modalità eval con testa a pesi casuali.

    La testa parte con pesi a zero (il modello coincide con la persistenza e
    non dipende dagli input): per i test che verificano DA COSA dipendono le
    predizioni le diamo pesi casuali, come dopo qualche passo di training.
    """
    torch.manual_seed(0)
    model = ResidualMLPBaseline(num_features=num_features, **kwargs)
    torch.nn.init.normal_(model.head.proj.weight, std=0.1)
    return model.eval()


class TestPersistenceModel(unittest.TestCase):
    def test_repeats_last_obs_for_every_step_and_quantile(self) -> None:
        batch = make_dummy_batch(batch_size=5)
        model = PersistenceModel()
        preds = model(batch)

        validate_predictions(preds, batch_size=5, require_monotonic_quantiles=True)
        expected = batch["last_obs"][:, None, :, None].expand(-1, 3, -1, 3)
        torch.testing.assert_close(preds, expected)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 0)


class TestResidualMLPBaseline(unittest.TestCase):
    def test_contract_and_monotonic_quantiles(self) -> None:
        batch = make_dummy_batch(batch_size=8)
        preds = _make_mlp()(batch)
        validate_predictions(preds, batch_size=8, require_monotonic_quantiles=True)

    def test_untrained_model_equals_persistence(self) -> None:
        # Pesi della testa a zero: mediana = last_obs, come la persistenza.
        batch = make_dummy_batch(batch_size=4)
        model = ResidualMLPBaseline(num_features=N_DEFAULT_FEATURES).eval()
        median = model(batch)[..., 1]
        torch.testing.assert_close(median, PersistenceModel()(batch)[..., 1])

    def test_default_and_ablated_feature_counts(self) -> None:
        for num_features in (N_DEFAULT_FEATURES, N_DEFAULT_FEATURES - 1):
            batch = make_dummy_batch(batch_size=4, num_features=num_features)
            preds = _make_mlp(num_features=num_features)(batch)
            validate_predictions(preds, batch_size=4, require_monotonic_quantiles=True)

        # Numero di feature del batch diverso da quello del modello -> errore chiaro.
        with self.assertRaises(ValueError):
            _make_mlp(num_features=N_DEFAULT_FEATURES)(make_dummy_batch(num_features=N_DEFAULT_FEATURES - 1))

    def test_long_sequences_160_days(self) -> None:
        batch = make_dummy_batch(
            batch_size=3, min_hist_len=150, max_hist_len=160, min_fut_len=150, max_fut_len=160
        )
        validate_predictions(_make_mlp()(batch), batch_size=3, require_monotonic_quantiles=True)

    def test_metadata_and_targets_do_not_affect_predictions(self) -> None:
        # Zero leakage: area, climate, sample_idx e i target non devono influire.
        batch = make_dummy_batch(batch_size=6)
        model = _make_mlp()
        reference = model(batch)

        altered = dict(batch)
        altered["area"] = ["ALTRA_AREA"] * 6
        altered["climate"] = ["unknown"] * 6
        altered["sample_idx"] = batch["sample_idx"] + 1000
        altered["target"] = torch.randn_like(batch["target"]) * 10.0
        altered["target_raw"] = torch.rand_like(batch["target_raw"])
        torch.testing.assert_close(model(altered), reference)

    def test_step1_ignores_future_after_its_target(self) -> None:
        # Causalità: cambiando il meteo futuro DOPO il giorno del 1° target,
        # la predizione dello step 1 non cambia, quella dello step 3 sì.
        batch = make_dummy_batch(batch_size=8, seed=1)
        model = _make_mlp()
        reference = model(batch)

        first_target = batch["future_target_positions"][:, 0]                    # (B,)
        days = torch.arange(batch["future"].shape[1]).unsqueeze(0)              # (1, T_f)
        after_first = (days > first_target.unsqueeze(1)).unsqueeze(-1)          # (B, T_f, 1)
        change = after_first & ~batch["future_mask"]                            # solo valori osservati

        altered = dict(batch)
        altered["future"] = torch.where(change, batch["future"] + 5.0, batch["future"])
        preds = model(altered)

        torch.testing.assert_close(preds[:, 0], reference[:, 0])
        self.assertFalse(torch.allclose(preds[:, 2], reference[:, 2]))

    def test_works_with_fully_masked_future(self) -> None:
        # Caso estremo (più forte dell'ablation use_future_meteo: false): nessun
        # dato futuro osservato. Le medie su finestre vuote non devono dare NaN.
        batch = make_dummy_batch(batch_size=4)
        altered = dict(batch)
        altered["future_mask"] = torch.ones_like(batch["future_mask"])
        altered["future"] = torch.zeros_like(batch["future"])
        validate_predictions(_make_mlp()(altered), batch_size=4, require_monotonic_quantiles=True)

    def test_loss_decreases_when_overfitting_one_batch(self) -> None:
        # Sanity check: su un solo batch la rete deve riuscire a ridurre molto la loss.
        torch.manual_seed(0)
        batch = make_dummy_batch(batch_size=16, seed=3)
        model = ResidualMLPBaseline(num_features=N_DEFAULT_FEATURES, dropout=0.0)
        criterion = PinballLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        with torch.no_grad():
            initial = criterion(model(batch), batch["target"], batch["target_mask"]).item()
        for _ in range(150):
            optimizer.zero_grad()
            loss = criterion(model(batch), batch["target"], batch["target_mask"])
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), 0.5 * initial)


class TestBuildModel(unittest.TestCase):
    def test_builds_model_from_config(self) -> None:
        cfg = load_config()
        model = build_model(cfg)
        self.assertIsInstance(model, ResidualMLPBaseline)
        self.assertEqual(model.num_features, N_DEFAULT_FEATURES)
        # I parametri del blocco model.params.residual_mlp vengono applicati.
        self.assertEqual(model.mlp[0].out_features, cfg["model"]["params"]["residual_mlp"]["hidden_dim"])

        cfg_persistence = copy.deepcopy(cfg)
        cfg_persistence["model"]["name"] = "persistence"
        self.assertIsInstance(build_model(cfg_persistence), PersistenceModel)

    def test_every_registered_model_respects_the_contract(self) -> None:
        cfg = copy.deepcopy(load_config())
        batch = make_dummy_batch(batch_size=4)
        for name in MODEL_REGISTRY:
            cfg["model"]["name"] = name
            preds = build_model(cfg).eval()(batch)
            validate_predictions(preds, batch_size=4, require_monotonic_quantiles=True)

    def test_drop_features_changes_num_features(self) -> None:
        self.assertEqual(infer_num_features({}), N_DEFAULT_FEATURES)
        self.assertEqual(infer_num_features({"preprocessing": {"drop_features": ["wind"]}}), N_DEFAULT_FEATURES - 1)
        with self.assertRaises(ValueError):
            infer_num_features({"preprocessing": {"drop_features": ["vento"]}})

        cfg = copy.deepcopy(load_config())
        cfg["preprocessing"]["drop_features"] = ["wind"]
        model = build_model(cfg).eval()
        batch = make_dummy_batch(batch_size=4, num_features=N_DEFAULT_FEATURES - 1)
        validate_predictions(model(batch), batch_size=4)

    def test_configuration_errors_are_explicit(self) -> None:
        cfg = load_config()
        for broken in ({"model": {"name": None}}, {"model": {"name": "modello_inesistente"}}):
            with self.assertRaises(ValueError):
                build_model(broken)

        cfg_typo = copy.deepcopy(cfg)
        cfg_typo["model"]["params"]["residual_mlp"]["hiden_dim"] = 64  # refuso
        with self.assertRaises(ValueError):
            build_model(cfg_typo)


class TestGatherTimeSteps(unittest.TestCase):
    def test_picks_the_requested_day_for_each_sample(self) -> None:
        x = torch.arange(2 * 5 * 3, dtype=torch.float32).view(2, 5, 3)  # (B=2, T=5, D=3)
        positions = torch.tensor([[0, 2], [4, 1]])
        out = gather_time_steps(x, positions)
        self.assertEqual(tuple(out.shape), (2, 2, 3))
        torch.testing.assert_close(out[0, 1], x[0, 2])
        torch.testing.assert_close(out[1, 0], x[1, 4])
        with self.assertRaises(ValueError):
            gather_time_steps(x[0], positions)


if __name__ == "__main__":
    unittest.main()
