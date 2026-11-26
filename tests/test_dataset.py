"""Test unitari e di integrazione per Dataset, collate_fn e DataLoader (src/dataset.py)."""

from __future__ import annotations

import copy
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import PROJECT_ROOT, load_config
from src.contract import (
    FEATURE_NAMES,
    METEO_CLOUD_FEATURES,
    N_DEFAULT_FEATURES,
    N_FUTURE,
    N_TARGETS,
    TARGET_HISTORY_FEATURES,
    TARGET_NAMES,
    validate_batch,
)
from src.dataset import (
    PreprocessingConfig,
    VegetationCollateFn,
    VegetationDataset,
    build_dataloaders,
    build_dataset,
    collate_vegetation_batch,
)
from src.losses import PinballLoss
from src.models.common import MonotonicQuantileHead, build_step_features, step_feature_dim
from src.preprocessing import ArcsinhScaler


def _make_dummy_scaler() -> ArcsinhScaler:
    mean = np.linspace(-1.0, 5.0, N_DEFAULT_FEATURES)
    var = np.linspace(0.5, 10.0, N_DEFAULT_FEATURES)
    target_mean = np.array([0.45, 0.50, 0.32])
    target_var = np.array([0.03, 0.015, 0.025])
    return ArcsinhScaler(mean, var, target_mean, target_var, eps=1e-5)


def _write_synthetic_area(
    directory: Path,
    area_name: str,
    *,
    hist_lens: list[int],
    fut_lens: list[int],
    climate: str = "Csa",
    seed: int = 0,
    corrupt_future_noise: bool = False,
) -> dict[str, np.ndarray]:
    """Crea una coppia `<area>.npz` + `<area>.json` fedele allo schema reale."""
    rng = np.random.default_rng(seed)
    n = len(hist_lens)
    assert len(fut_lens) == n

    h_arr = np.empty(n, dtype=object)
    hm_arr = np.empty(n, dtype=object)
    f_arr = np.empty(n, dtype=object)
    fm_arr = np.empty(n, dtype=object)
    fn_arr = np.empty(n, dtype=object)
    targets = np.empty((n, N_FUTURE, N_TARGETS), dtype=np.float32)
    target_masks = np.zeros((n, N_FUTURE, N_TARGETS), dtype=bool)
    positions = np.empty((n, N_FUTURE), dtype=np.int64)
    climates = np.array([climate] * n)

    for i in range(n):
        lh = hist_lens[i]
        lf = fut_lens[i]

        h = rng.normal(loc=1.0, scale=2.0, size=(lh, N_DEFAULT_FEATURES)).astype(np.float32)
        hm = rng.random((lh, N_DEFAULT_FEATURES)) < 0.08
        # Indici vegetativi (18..20): solo 3 clear-sky (inizio, metà, fine)
        hm[:, 18:21] = True
        hm[0, 18:21] = False
        hm[lh // 2, 18:21] = False
        hm[lh - 1, 18:21] = False
        h[0, 18:21] = rng.uniform(0.1, 0.8, size=3).astype(np.float32)
        h[lh // 2, 18:21] = rng.uniform(0.1, 0.8, size=3).astype(np.float32)
        h[lh - 1, 18:21] = rng.uniform(0.1, 0.8, size=3).astype(np.float32)
        h[hm] = np.nan

        f = rng.normal(loc=1.0, scale=2.0, size=(lf, N_DEFAULT_FEATURES)).astype(np.float32)
        fm = rng.random((lf, N_DEFAULT_FEATURES)) < 0.08
        # In future i target sono sempre NaN
        fm[:, 18:21] = True
        # Calendario/Fourier (11..17) sempre osservati
        fm[:, 11:18] = False
        f[fm] = np.nan

        fn = (
            np.full_like(f, 9999.0)
            if corrupt_future_noise
            else rng.normal(0.0, 0.1, size=f.shape).astype(np.float32)
        )

        p1 = max(0, lf // 3 - 1)
        p2 = max(p1 + 1, (2 * lf) // 3 - 1)
        p3 = lf - 1
        positions[i] = [p1, p2, p3]
        targets[i] = rng.uniform(0.05, 0.85, size=(N_FUTURE, N_TARGETS)).astype(np.float32)

        h_arr[i] = h
        hm_arr[i] = hm
        f_arr[i] = f
        fm_arr[i] = fm
        fn_arr[i] = fn

    np.savez(
        directory / f"{area_name}.npz",
        history=h_arr,
        history_mask=hm_arr,
        future=f_arr,
        future_mask=fm_arr,
        future_noise=fn_arr,
        target=targets,
        target_mask=target_masks,
        future_target_positions=positions,
        climate=climates,
    )

    meta = {
        "area": area_name,
        "feature_names": list(FEATURE_NAMES),
        "target_names": list(TARGET_NAMES),
        "metadata_records": [{"area": area_name, "climate": climate} for _ in range(n)],
    }
    with open(directory / f"{area_name}.json", "w", encoding="utf-8") as fp:
        json.dump(meta, fp)

    return {"history": h_arr, "history_mask": hm_arr, "future": f_arr, "future_mask": fm_arr, "target": targets}


class TestPreprocessingConfig(unittest.TestCase):
    def test_from_config_defaults(self) -> None:
        cfg = load_config()
        prep = PreprocessingConfig.from_config(cfg)
        self.assertTrue(prep.normalize_features)
        self.assertTrue(prep.normalize_targets)
        self.assertEqual(prep.nan_fill_value, 0.0)
        self.assertTrue(prep.mask_true_is_missing)
        self.assertTrue(prep.dynamic_padding)
        self.assertFalse(prep.use_climate)
        self.assertTrue(prep.use_future_meteo)
        self.assertEqual(prep.num_features, N_DEFAULT_FEATURES)

    def test_rejects_invalid_invariants(self) -> None:
        with self.assertRaises(ValueError):
            PreprocessingConfig(mask_true_is_missing=False)
        with self.assertRaises(ValueError):
            PreprocessingConfig(use_climate=True)
        with self.assertRaises(ValueError):
            PreprocessingConfig(drop_features=("feature_inesistente",))
        with self.assertRaises(ValueError):
            PreprocessingConfig(max_seq_len=2)


class TestSyntheticDatasetAndCollate(unittest.TestCase):
    def test_scaling_no_interpolation_and_dynamic_padding(self) -> None:
        scaler = _make_dummy_scaler()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_a = _write_synthetic_area(
                root, "AREA_A", hist_lens=[11, 20, 15], fut_lens=[15, 28, 19], climate="Csa", seed=1,
                corrupt_future_noise=True,
            )
            _write_synthetic_area(
                root, "AREA_B", hist_lens=[12, 35], fut_lens=[16, 40], climate="Dfb", seed=2,
            )

            ds = VegetationDataset(root, scaler=scaler, preprocessing=PreprocessingConfig())
            self.assertEqual(len(ds), 5)
            self.assertEqual(ds.num_features, N_DEFAULT_FEATURES)

            # Verifica singolo campione: NaN mai interpolati, maschera fedele, scaling corretto
            s0 = ds[0]
            raw_h0 = raw_a["history"][0]
            raw_hm0 = raw_a["history_mask"][0]
            self.assertTrue(torch.equal(s0["history_mask"], torch.from_numpy(raw_hm0)))
            # Posizioni mancanti riempite con 0.0
            self.assertTrue((s0["history"][s0["history_mask"]] == 0.0).all().item())
            # Posizioni valide uguali esattamente ad ArcsinhScaler(raw)
            expected_h0 = scaler.transform_features(raw_h0)
            valid0 = ~raw_hm0
            np.testing.assert_allclose(
                s0["history"].numpy()[valid0], expected_h0[valid0], rtol=1e-5, atol=1e-5
            )
            # In future le colonne target sono 100% mascherate e a 0.0
            self.assertTrue(s0["future_mask"][:, 18:21].all().item())
            self.assertTrue((s0["future"][:, 18:21] == 0.0).all().item())
            # future_noise corrotto non ha contaminato future
            raw_f0 = raw_a["future"][0]
            raw_fm0 = raw_a["future_mask"][0]
            expected_f0 = scaler.transform_features(raw_f0)
            f_valid0 = ~raw_fm0
            np.testing.assert_allclose(
                s0["future"].numpy()[f_valid0], expected_f0[f_valid0], rtol=1e-5, atol=1e-5
            )
            # last_obs e target sono nella stessa scala normalizzata, target_raw è grezzo
            expected_last_obs = scaler.transform_targets(raw_h0[-1, 18:21])
            np.testing.assert_allclose(s0["last_obs"].numpy(), expected_last_obs, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(s0["target_raw"].numpy(), raw_a["target"][0], rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(
                scaler.inverse_transform_targets(s0["target"]).numpy(),
                raw_a["target"][0],
                rtol=1e-5,
                atol=1e-5,
            )

            # Verifica DataLoader + collate_fn dinamico + validate_batch
            collate_fn = VegetationCollateFn.from_preprocessing(ds.preprocessing)
            # Deve essere picklable per num_workers > 0
            self.assertIsInstance(pickle.loads(pickle.dumps(collate_fn)), VegetationCollateFn)

            loader = DataLoader(ds, batch_size=3, shuffle=False, collate_fn=collate_fn)
            batches = list(loader)
            self.assertEqual(len(batches), 2)

            # Primo batch (AREA_A): max hist=20, max fut=28
            b0 = batches[0]
            validate_batch(b0, expected_num_features=N_DEFAULT_FEATURES)
            self.assertEqual(tuple(b0["history"].shape), (3, 20, N_DEFAULT_FEATURES))
            self.assertEqual(tuple(b0["future"].shape), (3, 28, N_DEFAULT_FEATURES))
            self.assertEqual(b0["area"], ["AREA_A", "AREA_A", "AREA_A"])
            self.assertEqual(b0["sample_idx"].tolist(), [0, 1, 2])

            # Secondo batch (AREA_B): max hist=35, max fut=40
            b1 = batches[1]
            validate_batch(b1, expected_num_features=N_DEFAULT_FEATURES)
            self.assertEqual(tuple(b1["history"].shape), (2, 35, N_DEFAULT_FEATURES))
            self.assertEqual(tuple(b1["future"].shape), (2, 40, N_DEFAULT_FEATURES))
            self.assertEqual(b1["sample_idx"].tolist(), [3, 4])

    def test_ablations_drop_features_no_future_meteo_fixed_padding_no_delta_t(self) -> None:
        scaler = _make_dummy_scaler()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_a = _write_synthetic_area(
                root, "AREA_A", hist_lens=[11, 18], fut_lens=[15, 22], climate="BSk", seed=7
            )

            # 1. drop_features (incluso un target storico!) + use_future_meteo=False + add_delta_t=False
            prep = PreprocessingConfig(
                use_future_meteo=False,
                drop_features=("wind", TARGET_HISTORY_FEATURES[0]),
                dynamic_padding=False,
                max_seq_len=30,
                add_delta_t=False,
            )
            ds = VegetationDataset(root, scaler=scaler, preprocessing=prep)
            self.assertEqual(ds.num_features, N_DEFAULT_FEATURES - 2)

            batch = collate_vegetation_batch(
                [ds[0], ds[1]],
                dynamic_padding=prep.dynamic_padding,
                max_seq_len=prep.max_seq_len,
                nan_fill_value=prep.nan_fill_value,
                add_delta_t=prep.add_delta_t,
            )
            validate_batch(batch, expected_num_features=N_DEFAULT_FEATURES - 2, check_delta_t_values=False)
            self.assertEqual(tuple(batch["history"].shape), (2, 30, N_DEFAULT_FEATURES - 2))
            self.assertEqual(tuple(batch["future"].shape), (2, 30, N_DEFAULT_FEATURES - 2))

            # delta_t tutti 0.0 quando add_delta_t=False
            self.assertTrue((batch["history_delta_t"] == 0.0).all().item())
            self.assertTrue((batch["future_delta_t"] == 0.0).all().item())
            self.assertTrue((batch["target_delta_t"] == 0.0).all().item())

            # Tutte le colonne meteo/nuvole rimaste in future devono essere mascherate e a 0.0
            meteo_idx = [i for i, n in enumerate(ds.feature_names) if n in METEO_CLOUD_FEATURES]
            self.assertGreater(len(meteo_idx), 0)
            self.assertTrue(batch["future_mask"][:, :, meteo_idx].all().item())
            self.assertTrue((batch["future"][:, :, meteo_idx] == 0.0).all().item())

            # Le colonne calendario/Fourier in future sui passi validi restano False (valide)
            cal_idx = [i for i, n in enumerate(ds.feature_names) if n.startswith("day_of_year")]
            self.assertFalse(batch["future_mask"][0, :15, cal_idx].any().item())

            # last_obs è stato comunque estratto da tutti e 3 gli indici PRIMA del drop
            expected_last_0 = scaler.transform_targets(raw_a["history"][0][-1, 18:21])
            np.testing.assert_allclose(batch["last_obs"][0].numpy(), expected_last_0, rtol=1e-5, atol=1e-5)

            # 2. Fixed padding rifiuta sequenze più lunghe di max_seq_len
            with self.assertRaises(ValueError):
                collate_vegetation_batch([ds[0], ds[1]], dynamic_padding=False, max_seq_len=20)

    def test_area_and_index_filtering(self) -> None:
        scaler = _make_dummy_scaler()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_synthetic_area(root, "AREA_1", hist_lens=[11, 12, 13], fut_lens=[15, 16, 17], seed=10)
            _write_synthetic_area(root, "AREA_2", hist_lens=[14, 15], fut_lens=[18, 19], seed=11)

            ds_area2 = VegetationDataset(root, scaler=scaler, areas=["AREA_2"])
            self.assertEqual(len(ds_area2), 2)
            self.assertEqual([ds_area2[i]["sample_idx"] for i in range(2)], [3, 4])
            self.assertEqual([ds_area2[i]["area"] for i in range(2)], ["AREA_2", "AREA_2"])

            ds_idx = VegetationDataset(root, scaler=scaler, indices=[0, 2, 4])
            self.assertEqual(len(ds_idx), 3)
            self.assertEqual([ds_idx[i]["sample_idx"] for i in range(3)], [0, 2, 4])
            self.assertEqual([ds_idx[i]["area"] for i in range(3)], ["AREA_1", "AREA_1", "AREA_2"])

    def test_end_to_end_with_common_components_and_loss(self) -> None:
        scaler = _make_dummy_scaler()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_synthetic_area(root, "AREA_1", hist_lens=[11, 17, 25, 14], fut_lens=[15, 20, 30, 18], seed=99)
            ds = VegetationDataset(root, scaler=scaler)
            loader = DataLoader(ds, batch_size=4, collate_fn=VegetationCollateFn())
            batch = next(iter(loader))
            validate_batch(batch, expected_num_features=N_DEFAULT_FEATURES)

            # Passa attraverso build_step_features -> lineare -> MonotonicQuantileHead -> PinballLoss
            d_in = step_feature_dim(ds.num_features)
            proj = torch.nn.Linear(d_in, 16)
            head = MonotonicQuantileHead(in_features=16, residual=True)
            loss_fn = PinballLoss()

            fut_in = build_step_features(
                batch["future"], batch["future_mask"], batch["future_delta_t"], batch["future_padding_mask"]
            )
            h_fut = proj(fut_in)  # (B, T_f, 16)
            idx = batch["future_target_positions"].unsqueeze(-1).expand(-1, -1, 16)
            target_repr = h_fut.gather(dim=1, index=idx)  # (B, 3, 16)
            preds = head(target_repr, last_obs=batch["last_obs"])

            loss = loss_fn(preds, batch["target"], batch["target_mask"])
            loss.backward()
            self.assertTrue(torch.isfinite(loss).item())

            preds_raw = scaler.inverse_transform_predictions(preds.detach())
            self.assertTrue(torch.isfinite(preds_raw).all().item())
            self.assertTrue((preds_raw[..., 0] <= preds_raw[..., 1]).all().item())
            self.assertTrue((preds_raw[..., 1] <= preds_raw[..., 2]).all().item())


class TestRealDataLoaders(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config()
        self.data_root = PROJECT_ROOT / self.cfg["data"]["root"]
        if not (self.data_root / self.cfg["data"]["splits"]["train"]).is_dir():
            self.skipTest("DATA/ non presente nel workspace.")

    def test_real_id_and_ood_t_dataloaders(self) -> None:
        loaders = build_dataloaders(
            self.cfg,
            splits=("id", "ood_t"),
            batch_size=64,
            num_workers=0,
        )
        self.assertEqual(len(loaders["id"].dataset), 3336)
        self.assertEqual(len(loaders["ood_t"].dataset), 6511)

        for name, loader in loaders.items():
            for step, batch in enumerate(loader):
                validate_batch(batch, expected_num_features=N_DEFAULT_FEATURES)
                if step >= 2:
                    break

    def test_real_ood_s_long_sequences_and_unknown_climate(self) -> None:
        ds_ood_s = build_dataset("ood_s", self.cfg)
        self.assertEqual(len(ds_ood_s), 25576)
        loader = DataLoader(ds_ood_s, batch_size=128, shuffle=False, collate_fn=VegetationCollateFn())
        batch = next(iter(loader))
        validate_batch(batch, expected_num_features=N_DEFAULT_FEATURES)
        self.assertTrue(all(c == "unknown" for c in batch["climate"]))

    def test_real_train_val_area_split_dataloaders(self) -> None:
        loaders = build_dataloaders(
            self.cfg,
            splits=("train", "val"),
            batch_size=128,
            num_workers=0,
        )
        ds_tr = loaders["train"].dataset
        ds_va = loaders["val"].dataset
        self.assertEqual(len(ds_tr) + len(ds_va), 104699)

        # Nessuna sovrapposizione di aree tra train e val
        tr_areas = set(ds_tr._areas)  # type: ignore[attr-defined]
        va_areas = set(ds_va._areas)  # type: ignore[attr-defined]
        self.assertEqual(len(tr_areas & va_areas), 0)

        for split_name in ("train", "val"):
            batch = next(iter(loaders[split_name]))
            validate_batch(batch, expected_num_features=N_DEFAULT_FEATURES)

    def test_real_train_val_random_split(self) -> None:
        cfg_rand = copy.deepcopy(self.cfg)
        cfg_rand["data"]["val_split_by"] = "random"
        # Verifica solo lo split di validazione random (15% dei campioni) per velocità
        ds_va = build_dataset("val", cfg_rand)
        self.assertEqual(len(ds_va), int(round(104699 * 0.15)))
        batch = collate_vegetation_batch([ds_va[0], ds_va[1], ds_va[-1]])
        validate_batch(batch, expected_num_features=N_DEFAULT_FEATURES)


if __name__ == "__main__":
    unittest.main()
