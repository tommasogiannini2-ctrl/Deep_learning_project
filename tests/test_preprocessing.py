"""Test unitari per scaler standardization_arcsinh e split train/val (src/preprocessing.py)."""

from __future__ import annotations

import copy
import unittest

import numpy as np
import torch

from src.config import PROJECT_ROOT, load_config
from src.contract import N_DEFAULT_FEATURES, N_FUTURE, N_QUANTILES, N_TARGETS
from src.preprocessing import (
    ArcsinhScaler,
    AreaInfo,
    make_train_val_split,
    split_areas_balanced,
    split_samples_random,
)


def _make_dummy_scaler() -> ArcsinhScaler:
    mean = np.linspace(-2.0, 10.0, N_DEFAULT_FEATURES)
    var = np.linspace(0.5, 25.0, N_DEFAULT_FEATURES)
    target_mean = np.array([0.45, 0.50, 0.32])
    target_var = np.array([0.03, 0.015, 0.025])
    return ArcsinhScaler(mean, var, target_mean, target_var, eps=1e-5)


class TestArcsinhScaler(unittest.TestCase):
    def test_roundtrip_features_numpy_and_torch(self) -> None:
        scaler = _make_dummy_scaler()
        rng = np.random.default_rng(0)
        x_np = rng.normal(loc=5.0, scale=10.0, size=(4, 15, N_DEFAULT_FEATURES)).astype(np.float32)

        y_np = scaler.transform_features(x_np)
        x_rec_np = scaler.inverse_transform_features(y_np)
        self.assertEqual(y_np.dtype, np.float32)
        np.testing.assert_allclose(x_rec_np, x_np, rtol=1e-5, atol=1e-5)

        x_t = torch.from_numpy(x_np)
        y_t = scaler.transform_features(x_t)
        x_rec_t = scaler.inverse_transform_features(y_t)
        self.assertEqual(y_t.dtype, torch.float32)
        self.assertTrue(torch.allclose(y_t, torch.from_numpy(y_np), rtol=1e-5, atol=1e-5))
        self.assertTrue(torch.allclose(x_rec_t, x_t, rtol=1e-5, atol=1e-5))

    def test_nan_preserved_without_interpolation(self) -> None:
        scaler = _make_dummy_scaler()
        x = np.ones((2, 5, N_DEFAULT_FEATURES), dtype=np.float32)
        x[0, 2, 0] = np.nan
        x[1, 4, 18:] = np.nan

        y = scaler.transform_features(x)
        self.assertTrue(np.array_equal(np.isnan(y), np.isnan(x)))

        y_t = scaler.transform_features(torch.from_numpy(x))
        self.assertTrue(torch.equal(torch.isnan(y_t), torch.from_numpy(np.isnan(x))))

    def test_feature_subset_for_drop_features(self) -> None:
        scaler = _make_dummy_scaler()
        rng = np.random.default_rng(1)
        x_full = rng.normal(size=(3, 7, N_DEFAULT_FEATURES)).astype(np.float32)
        keep_idx = list(range(1, N_DEFAULT_FEATURES))  # es. drop "wind" (indice 0)

        y_full = scaler.transform_features(x_full)
        y_sub = scaler.transform_features(x_full[..., keep_idx], feature_indices=keep_idx)
        np.testing.assert_allclose(y_sub, y_full[..., keep_idx], rtol=1e-6, atol=1e-6)

        x_sub_rec = scaler.inverse_transform_features(y_sub, feature_indices=keep_idx)
        np.testing.assert_allclose(x_sub_rec, x_full[..., keep_idx], rtol=1e-5, atol=1e-5)

    def test_targets_and_predictions_inverse_axis_and_monotonicity(self) -> None:
        scaler = _make_dummy_scaler()
        # (B=4, H=3, C=3)
        y_raw = torch.tensor(
            [[[0.2, 0.3, 0.1], [0.4, 0.5, 0.2], [0.6, 0.7, 0.3]]] * 4,
            dtype=torch.float32,
        )
        y_scaled = scaler.transform_targets(y_raw)
        y_rec = scaler.inverse_transform_targets(y_scaled)
        self.assertTrue(torch.allclose(y_rec, y_raw, rtol=1e-5, atol=1e-5))

        # Predizioni (B, H=3, C=3, Q=3) con quantili ordinati q10 < q50 < q90
        preds_scaled = torch.stack([y_scaled - 0.25, y_scaled, y_scaled + 0.25], dim=-1)
        self.assertEqual(tuple(preds_scaled.shape), (4, N_FUTURE, N_TARGETS, N_QUANTILES))

        preds_raw = scaler.inverse_transform_predictions(preds_scaled)
        # La mediana (q=1) deve coincidere esattamente con y_raw
        self.assertTrue(torch.allclose(preds_raw[..., 1], y_raw, rtol=1e-5, atol=1e-5))
        # sinh è strettamente crescente -> monotonicità preservata
        self.assertTrue((preds_raw[..., 0] < preds_raw[..., 1]).all().item())
        self.assertTrue((preds_raw[..., 1] < preds_raw[..., 2]).all().item())

    def test_validation_errors(self) -> None:
        with self.assertRaises(ValueError):
            ArcsinhScaler(
                np.zeros(N_DEFAULT_FEATURES),
                np.ones(N_DEFAULT_FEATURES),
                np.zeros(N_TARGETS),
                np.ones(N_TARGETS),
                mode="minmax",
            )
        with self.assertRaises(ValueError):
            ArcsinhScaler(
                np.zeros(N_DEFAULT_FEATURES),
                -np.ones(N_DEFAULT_FEATURES),
                np.zeros(N_TARGETS),
                np.ones(N_TARGETS),
            )
        scaler = _make_dummy_scaler()
        with self.assertRaises(ValueError):
            scaler.transform_features(np.zeros((2, 5, 10), dtype=np.float32))
        with self.assertRaises(ValueError):
            scaler.inverse_transform_predictions(torch.zeros((2, 3, 4, 3)))

    def test_from_config_real_train_scaler(self) -> None:
        cfg = load_config()
        scaler_path = PROJECT_ROOT / cfg["data"]["root"] / cfg["data"]["scaler_file"]
        if not scaler_path.is_file():
            self.skipTest("DATA/ non presente nel workspace.")
        scaler = ArcsinhScaler.from_config(cfg)
        self.assertEqual(scaler.mode, "standardization_arcsinh")
        self.assertAlmostEqual(scaler.eps, 1e-5)
        self.assertEqual(scaler.mean.shape, (N_DEFAULT_FEATURES,))
        self.assertEqual(scaler.target_mean.shape, (N_TARGETS,))


class TestTrainValSplit(unittest.TestCase):
    def _synthetic_areas(self) -> list[AreaInfo]:
        return [
            AreaInfo("A01", 1000, {"Csa": 1000}),
            AreaInfo("A02", 600, {"Csa": 600}),
            AreaInfo("A03", 300, {"Csa": 300}),
            AreaInfo("A04", 100, {"Csa": 100}),
            AreaInfo("B01", 800, {"BSk": 800}),
            AreaInfo("B02", 400, {"BSk": 400}),
            AreaInfo("B03", 200, {"BSk": 200}),
            AreaInfo("C01", 500, {"Dfb": 500}),
            AreaInfo("C02", 300, {"Dfb": 300}),
            AreaInfo("C03", 150, {"Dfb": 150}),
            AreaInfo("C04", 50, {"Dfb": 50}),
            # Clima presente in una sola area: deve restare sempre nel train
            AreaInfo("D01", 120, {"Dfa": 120}),
        ]

    def test_balanced_area_split_synthetic(self) -> None:
        areas = self._synthetic_areas()
        split1 = split_areas_balanced(areas, val_fraction=0.15, balance_climate=True, seed=42)
        split2 = split_areas_balanced(areas, val_fraction=0.15, balance_climate=True, seed=42)

        # Riproducibilità esatta
        self.assertEqual(split1.train_areas, split2.train_areas)
        self.assertEqual(split1.val_areas, split2.val_areas)

        # Partizione completa senza overlap
        tr_set = set(split1.train_areas)
        va_set = set(split1.val_areas)
        self.assertEqual(len(tr_set & va_set), 0)
        self.assertEqual(tr_set | va_set, {a.area for a in areas})
        self.assertEqual(split1.total_samples, sum(a.n_samples for a in areas))

        # Clima con una sola area mai rimosso dal train
        self.assertIn("D01", tr_set)
        self.assertIn("Dfa", split1.train_climate_counts)
        self.assertNotIn("Dfa", split1.val_climate_counts)

        # Frazione val vicina al target 0.15
        self.assertAlmostEqual(split1.val_fraction_actual, 0.15, delta=0.03)

    def test_random_split_synthetic(self) -> None:
        areas = self._synthetic_areas()
        split = split_samples_random(areas, val_fraction=0.15, seed=42)
        self.assertEqual(split.mode, "random")
        self.assertIsNotNone(split.train_indices)
        self.assertIsNotNone(split.val_indices)
        assert split.train_indices is not None and split.val_indices is not None
        self.assertEqual(len(set(split.train_indices.tolist()) & set(split.val_indices.tolist())), 0)
        self.assertEqual(split.total_samples, sum(a.n_samples for a in areas))
        self.assertAlmostEqual(split.val_fraction_actual, 0.15, delta=0.005)

    def test_real_train_split_from_config(self) -> None:
        cfg = load_config()
        train_dir = PROJECT_ROOT / cfg["data"]["root"] / cfg["data"]["splits"]["train"]
        if not train_dir.is_dir():
            self.skipTest("DATA/ non presente nel workspace.")

        split = make_train_val_split(cfg)
        self.assertEqual(split.mode, "area")
        self.assertEqual(len(split.train_areas) + len(split.val_areas), 84)
        self.assertEqual(len(set(split.train_areas) & set(split.val_areas)), 0)
        self.assertEqual(split.total_samples, 104699)

        # Frazione complessiva molto vicina a 0.15
        self.assertAlmostEqual(split.val_fraction_actual, 0.15, delta=0.01)

        # Tutti i 9 climi restano presenti nel train (incluso Dfa con 1 sola area)
        self.assertEqual(len(split.train_climate_counts), 9)
        self.assertIn("Dfa", split.train_climate_counts)
        self.assertNotIn("Dfa", split.val_climate_counts)

        # I 4 climi principali (Csa, BSk, Dfb, Cfb) devono essere ben bilanciati vicino al 15%
        for clim in ("Csa", "BSk", "Dfb", "Cfb"):
            tr = split.train_climate_counts[clim]
            va = split.val_climate_counts.get(clim, 0)
            frac = va / (tr + va)
            self.assertGreater(frac, 0.10, f"Clima {clim} sottorappresentato in val: {frac:.3f}")
            self.assertLess(frac, 0.22, f"Clima {clim} sovrarappresentato in val: {frac:.3f}")

        # Verifica anche modalità random tramite config modificata in memoria
        cfg_rand = copy.deepcopy(cfg)
        cfg_rand["data"]["val_split_by"] = "random"
        split_rand = make_train_val_split(cfg_rand)
        self.assertEqual(split_rand.total_samples, 104699)
        self.assertAlmostEqual(split_rand.val_fraction_actual, 0.15, delta=0.001)


if __name__ == "__main__":
    unittest.main()
