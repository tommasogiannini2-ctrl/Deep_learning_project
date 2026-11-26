"""PyTorch Dataset, collate_fn con padding dinamico e DataLoader per GreenEarthNet.

Questo modulo implementa lo Step 4 della pipeline dati, rispettando al 100%
il contratto definito in `docs/batch_contract.md` e validato da `src/contract.py`:

- Caricamento efficiente di coppie `<area>.npz` / `<area>.json` in buffer NumPy
  contigui (stile CSR con array di offset) senza mai caricare `future_noise` né
  interpolare i `NaN`.
- Convenzione unica delle maschere (`True` = `NaN` originale, feature oscurata
  o timestep di padding; `False` = valore osservato valido).
- Applicazione opzionale di `ArcsinhScaler` (sempre stimato sul solo train)
  prima del riempimento con `nan_fill_value` (default `0.0`), garantendo
  assenza totale di `NaN`/`Inf` in tutti i tensori float.
- Right-padding dinamico per batch (o fisso a `max_seq_len`) con calcolo
  vettorizzato delle feature temporali continue `history_delta_t`,
  `future_delta_t` e `target_delta_t`.
- Costruzione centralizzata dei `DataLoader` per `train`, `val`, `id`, `ood_s`
  e `ood_t` a partire da `config.yaml`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import PROJECT_ROOT, load_config
from src.contract import (
    Batch,
    FEATURE_NAMES,
    METEO_CLOUD_FEATURES,
    N_DEFAULT_FEATURES,
    N_FUTURE,
    N_TARGETS,
    TARGET_HISTORY_FEATURES,
    TARGET_NAMES,
)
from src.preprocessing import ArcsinhScaler, TrainValSplit, make_train_val_split

# Indici canonici dei 3 indici vegetativi in history/future prima di drop_features
_TARGET_HISTORY_COL_INDICES: tuple[int, ...] = tuple(
    FEATURE_NAMES.index(name) for name in TARGET_HISTORY_FEATURES
)


# =============================================================================
# 1. Configurazione preprocessing (con validazione rigida delle invarianti)
# =============================================================================


@dataclass(frozen=True)
class PreprocessingConfig:
    """Parametri di preprocessing estratti da `config.yaml` (`preprocessing`)."""

    normalize_features: bool = True
    normalize_targets: bool = True
    nan_fill_value: float = 0.0
    mask_true_is_missing: bool = True
    max_seq_len: int = 150
    dynamic_padding: bool = True
    use_climate: bool = False
    use_future_meteo: bool = True
    drop_features: tuple[str, ...] = ()
    add_delta_t: bool = True

    def __post_init__(self) -> None:
        if not self.mask_true_is_missing:
            raise ValueError(
                "preprocessing.mask_true_is_missing deve essere True: la convenzione "
                "(True = mancante/padding, False = valido) è immutabile."
            )
        if self.use_climate:
            raise ValueError(
                "preprocessing.use_climate deve essere False: 'climate' vale 'unknown' "
                "nel 100% di ood_s ed è consentito solo a fini diagnostici."
            )
        if self.max_seq_len < N_FUTURE:
            raise ValueError(
                f"preprocessing.max_seq_len ({self.max_seq_len}) deve essere >= {N_FUTURE}."
            )
        if not np.isfinite(self.nan_fill_value):
            raise ValueError(
                f"preprocessing.nan_fill_value deve essere finito, trovato {self.nan_fill_value}."
            )
        unknown_features = [f for f in self.drop_features if f not in FEATURE_NAMES]
        if unknown_features:
            raise ValueError(
                f"Feature non riconosciute in preprocessing.drop_features: {unknown_features}. "
                f"Feature valide: {list(FEATURE_NAMES)}."
            )
        if len(set(self.drop_features)) >= N_DEFAULT_FEATURES:
            raise ValueError("Impossibile rimuovere tutte le feature in preprocessing.drop_features.")

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None = None) -> "PreprocessingConfig":
        """Costruisce `PreprocessingConfig` dal dizionario di `config.yaml`."""
        if cfg is None:
            cfg = load_config()
        prep = cfg.get("preprocessing", {})
        drop_raw = prep.get("drop_features") or []
        return cls(
            normalize_features=bool(prep.get("normalize_features", True)),
            normalize_targets=bool(prep.get("normalize_targets", True)),
            nan_fill_value=float(prep.get("nan_fill_value", 0.0)),
            mask_true_is_missing=bool(prep.get("mask_true_is_missing", True)),
            max_seq_len=int(prep.get("max_seq_len", 150)),
            dynamic_padding=bool(prep.get("dynamic_padding", True)),
            use_climate=bool(prep.get("use_climate", False)),
            use_future_meteo=bool(prep.get("use_future_meteo", True)),
            drop_features=tuple(str(f) for f in drop_raw),
            add_delta_t=bool(prep.get("add_delta_t", True)),
        )

    @property
    def kept_feature_indices(self) -> tuple[int, ...]:
        """Indici 0-based delle feature mantenute rispetto all'ordine canonico `FEATURE_NAMES`."""
        dropped = set(self.drop_features)
        return tuple(i for i, name in enumerate(FEATURE_NAMES) if name not in dropped)

    @property
    def kept_feature_names(self) -> tuple[str, ...]:
        """Nomi delle feature mantenute nell'ordine canonico."""
        return tuple(FEATURE_NAMES[i] for i in self.kept_feature_indices)

    @property
    def num_features(self) -> int:
        """Numero effettivo di feature `F` nei tensori `history` e `future`."""
        return len(self.kept_feature_indices)


# =============================================================================
# 2. Helper di estrazione sicura (senza interpolazione)
# =============================================================================


def _extract_last_obs(history_raw: np.ndarray) -> np.ndarray:
    """Estrae l'ultima osservazione clear-sky `[NDVI, GNDVI, EVI]` da `history_raw`.

    Per contratto nei 4 split forniti è sempre presente all'ultimo timestep `L_h - 1`.
    Come salvaguardia per lo split nascosto, se l'ultimo timestep contiene NaN si
    risale all'ultimo timestep osservato per ciascun indice (senza alcuna interpolazione).
    """
    target_cols = history_raw[:, _TARGET_HISTORY_COL_INDICES]  # (L_h, 3)
    last_row = target_cols[-1]
    if np.all(np.isfinite(last_row)):
        return last_row.astype(np.float32, copy=True)

    out = np.zeros(N_TARGETS, dtype=np.float32)
    for c in range(N_TARGETS):
        valid = np.flatnonzero(np.isfinite(target_cols[:, c]))
        if valid.size > 0:
            out[c] = float(target_cols[valid[-1], c])
    return out


def _validate_area_json(json_path: Path, expected_count: int) -> None:
    """Verifica coerenza metadati di `<area>.json` con il contratto canonico."""
    if not json_path.is_file():
        raise FileNotFoundError(f"File metadati JSON mancante: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feat_names = tuple(meta.get("feature_names", ()))
    if feat_names != FEATURE_NAMES:
        raise ValueError(
            f"feature_names in {json_path.name} non coincide con l'ordine canonico."
        )
    target_names = tuple(meta.get("target_names", ()))
    if target_names != TARGET_NAMES:
        raise ValueError(
            f"target_names in {json_path.name} non coincide con l'ordine canonico."
        )
    records = meta.get("metadata_records")
    if isinstance(records, list) and len(records) != expected_count:
        raise ValueError(
            f"Numero campioni incoerente in {json_path.name}: "
            f"{len(records)} nel JSON vs {expected_count} nell'NPZ."
        )


# =============================================================================
# 3. PyTorch Dataset (buffer contigui stile CSR, zero interpolazione)
# =============================================================================


class VegetationDataset(Dataset[dict[str, Any]]):
    """Dataset PyTorch per le serie temporali multi-output clear-sky di GreenEarthNet.

    Carica una sola volta i file `<area>.npz` e `<area>.json` richiesti in buffer
    NumPy contigui (CSR con offset), applicando normalizzazione `standardization_arcsinh`
    (solo dallo scaler del train), masking (`True` = mancante) e riempimento con
    `nan_fill_value` senza alcuna interpolazione e senza mai leggere `future_noise`.
    """

    def __init__(
        self,
        split_dir: str | Path,
        *,
        scaler: ArcsinhScaler | None = None,
        preprocessing: PreprocessingConfig | None = None,
        areas: Sequence[str] | None = None,
        indices: Sequence[int] | np.ndarray | None = None,
        verify_json: bool = True,
    ) -> None:
        self.split_dir = Path(split_dir)
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"Cartella split non trovata: {self.split_dir}")

        self.preprocessing = preprocessing or PreprocessingConfig()
        self.scaler = scaler

        if (self.preprocessing.normalize_features or self.preprocessing.normalize_targets) and self.scaler is None:
            raise ValueError(
                "È necessario passare `scaler` quando `normalize_features=True` o `normalize_targets=True`."
            )
        if areas is not None and indices is not None:
            raise ValueError("Specificare solo uno tra `areas` e `indices`, non entrambi.")

        all_npz_files = sorted(self.split_dir.glob("*.npz"))
        if not all_npz_files:
            raise ValueError(f"Nessun file .npz trovato in {self.split_dir}")

        area_filter: set[str] | None = set(areas) if areas is not None else None
        if area_filter is not None:
            available = {p.stem for p in all_npz_files}
            missing_areas = sorted(area_filter - available)
            if missing_areas:
                raise ValueError(
                    f"Aree richieste non presenti in {self.split_dir}: {missing_areas}"
                )

        requested_indices: np.ndarray | None = None
        if indices is not None:
            requested_indices = np.asarray(indices, dtype=np.int64)
            if requested_indices.ndim != 1:
                raise ValueError("`indices` deve essere un array/sequenza 1D.")
            if requested_indices.size == 0:
                raise ValueError("`indices` non può essere vuoto.")

        # Indici feature mantenuti e colonne speciali nello spazio post-drop
        kept_indices = self.preprocessing.kept_feature_indices
        self.feature_names: tuple[str, ...] = self.preprocessing.kept_feature_names
        self.num_features: int = self.preprocessing.num_features
        kept_idx_arr = np.asarray(kept_indices, dtype=np.int64)
        all_kept = len(kept_indices) == N_DEFAULT_FEATURES

        # Colonne target e meteo/nuvole nello spazio delle feature mantenute
        target_set = set(TARGET_HISTORY_FEATURES)
        meteo_cloud_set = set(METEO_CLOUD_FEATURES)
        future_target_cols = np.asarray(
            [i for i, name in enumerate(self.feature_names) if name in target_set],
            dtype=np.int64,
        )
        future_meteo_cols = np.asarray(
            [i for i, name in enumerate(self.feature_names) if name in meteo_cloud_set],
            dtype=np.int64,
        )

        # Accumulatori per costruire i buffer contigui finali
        hist_chunks: list[np.ndarray] = []
        hist_mask_chunks: list[np.ndarray] = []
        hist_lens: list[int] = []

        fut_chunks: list[np.ndarray] = []
        fut_mask_chunks: list[np.ndarray] = []
        fut_lens: list[int] = []

        target_pos_chunks: list[np.ndarray] = []
        last_obs_list: list[np.ndarray] = []
        target_raw_chunks: list[np.ndarray] = []
        target_mask_chunks: list[np.ndarray] = []
        sample_idx_chunks: list[np.ndarray] = []
        area_list: list[str] = []
        climate_list: list[str] = []

        # Se `indices` è fornito, mappiamo ciascun indice globale nella sua area
        sorted_req_indices: np.ndarray | None = None
        if requested_indices is not None:
            sorted_req_indices = np.sort(requested_indices)
            if sorted_req_indices[0] < 0:
                raise IndexError(f"Indice negativo non valido in `indices`: {sorted_req_indices[0]}")

        global_cursor = 0
        for npz_path in all_npz_files:
            area_name = npz_path.stem
            json_path = npz_path.with_suffix(".json")

            # Se filtriamo per `areas`, saltiamo le aree non richieste ma manteniamo
            # `global_cursor` allineato agli indici globali dello split leggendo solo il JSON.
            if area_filter is not None and area_name not in area_filter:
                if json_path.is_file():
                    with open(json_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    global_cursor += len(meta.get("metadata_records", []))
                else:
                    with np.load(npz_path, allow_pickle=True) as d:
                        global_cursor += int(d["target"].shape[0])
                continue

            with np.load(npz_path, allow_pickle=True) as d:
                n_area = int(d["target"].shape[0])
                area_start = global_cursor
                area_end = global_cursor + n_area
                global_cursor = area_end

                if sorted_req_indices is not None:
                    lo = int(np.searchsorted(sorted_req_indices, area_start, side="left"))
                    hi = int(np.searchsorted(sorted_req_indices, area_end, side="left"))
                    if lo == hi:
                        continue
                    local_sel = sorted_req_indices[lo:hi] - area_start
                    global_sel = sorted_req_indices[lo:hi]
                else:
                    local_sel = None
                    global_sel = np.arange(area_start, area_end, dtype=np.int64)

                if verify_json:
                    _validate_area_json(json_path, expected_count=n_area)

                # Carichiamo SOLO le chiavi consentite dal contratto (MAI future_noise!)
                h_obj = d["history"]
                hm_obj = d["history_mask"]
                f_obj = d["future"]
                fm_obj = d["future_mask"]
                t_raw = np.asarray(d["target"], dtype=np.float32)
                t_mask = np.asarray(d["target_mask"], dtype=bool)
                pos = np.asarray(d["future_target_positions"], dtype=np.int64)
                clim = d["climate"]

                if local_sel is not None:
                    h_iter = h_obj[local_sel]
                    hm_iter = hm_obj[local_sel]
                    f_iter = f_obj[local_sel]
                    fm_iter = fm_obj[local_sel]
                    t_raw = t_raw[local_sel]
                    t_mask = t_mask[local_sel]
                    pos = pos[local_sel]
                    clim = clim[local_sel]
                else:
                    h_iter = h_obj
                    hm_iter = hm_obj
                    f_iter = f_obj
                    fm_iter = fm_obj

                n_sel = len(t_raw)
                if n_sel == 0:
                    continue

                # Estrae last_obs PRIMA di qualsiasi drop_features
                area_last_obs = np.empty((n_sel, N_TARGETS), dtype=np.float32)
                area_h_list: list[np.ndarray] = []
                area_hm_list: list[np.ndarray] = []
                area_f_list: list[np.ndarray] = []
                area_fm_list: list[np.ndarray] = []

                for i in range(n_sel):
                    h_i = np.asarray(h_iter[i], dtype=np.float32)
                    hm_i = np.asarray(hm_iter[i], dtype=bool)
                    f_i = np.asarray(f_iter[i], dtype=np.float32)
                    fm_i = np.asarray(fm_iter[i], dtype=bool)

                    area_last_obs[i] = _extract_last_obs(h_i)

                    if not all_kept:
                        h_i = h_i[:, kept_idx_arr]
                        hm_i = hm_i[:, kept_idx_arr]
                        f_i = f_i[:, kept_idx_arr]
                        fm_i = fm_i[:, kept_idx_arr]

                    hist_lens.append(int(h_i.shape[0]))
                    fut_lens.append(int(f_i.shape[0]))
                    area_h_list.append(h_i)
                    area_hm_list.append(hm_i)
                    area_f_list.append(f_i)
                    area_fm_list.append(fm_i)

                # Concatena per area e applica normalizzazione/masking in modo vettorizzato
                h_cat = np.concatenate(area_h_list, axis=0)
                hm_cat = np.concatenate(area_hm_list, axis=0) | ~np.isfinite(h_cat)

                f_cat = np.concatenate(area_f_list, axis=0)
                fm_cat = np.concatenate(area_fm_list, axis=0) | ~np.isfinite(f_cat)

                # In future le colonne target sono SEMPRE mascherate (zero leakage)
                if future_target_cols.size > 0:
                    fm_cat[:, future_target_cols] = True
                # Ablation: oscura meteo+nuvole futuri se use_future_meteo=False
                if not self.preprocessing.use_future_meteo and future_meteo_cols.size > 0:
                    fm_cat[:, future_meteo_cols] = True

                if self.preprocessing.normalize_features:
                    assert self.scaler is not None
                    feat_idx_arg = None if all_kept else kept_indices
                    h_cat = self.scaler.transform_features(h_cat, feature_indices=feat_idx_arg)
                    f_cat = self.scaler.transform_features(f_cat, feature_indices=feat_idx_arg)
                    # Qualsiasi non-finito post-scaling resta marcato nella maschera
                    hm_cat |= ~np.isfinite(h_cat)
                    fm_cat |= ~np.isfinite(f_cat)

                fill_val = np.float32(self.preprocessing.nan_fill_value)
                h_cat = np.where(hm_cat, fill_val, h_cat).astype(np.float32, copy=False)
                f_cat = np.where(fm_cat, fill_val, f_cat).astype(np.float32, copy=False)

                hist_chunks.append(h_cat)
                hist_mask_chunks.append(hm_cat)
                fut_chunks.append(f_cat)
                fut_mask_chunks.append(fm_cat)

                # Target e maschera target (nessun NaN consentito nei tensori float)
                t_mask = t_mask | ~np.isfinite(t_raw)
                t_raw_clean = np.where(t_mask, fill_val, t_raw).astype(np.float32, copy=False)

                target_raw_chunks.append(t_raw_clean)
                target_mask_chunks.append(t_mask)
                target_pos_chunks.append(pos)
                last_obs_list.append(area_last_obs)
                sample_idx_chunks.append(global_sel)
                area_list.extend([area_name] * n_sel)
                climate_list.extend(str(c) for c in clim.tolist())

        if sorted_req_indices is not None and sorted_req_indices[-1] >= global_cursor:
            raise IndexError(
                f"Indice {sorted_req_indices[-1]} fuori range per split con {global_cursor} campioni."
            )

        if not hist_chunks:
            raise ValueError(f"Nessun campione caricato da {self.split_dir}.")

        # Assemblaggio buffer CSR finali
        self._history_values: np.ndarray = np.ascontiguousarray(
            np.concatenate(hist_chunks, axis=0), dtype=np.float32
        )
        self._history_masks: np.ndarray = np.ascontiguousarray(
            np.concatenate(hist_mask_chunks, axis=0), dtype=bool
        )
        self._history_offsets: np.ndarray = np.empty(len(hist_lens) + 1, dtype=np.int64)
        self._history_offsets[0] = 0
        np.cumsum(hist_lens, out=self._history_offsets[1:])

        self._future_values: np.ndarray = np.ascontiguousarray(
            np.concatenate(fut_chunks, axis=0), dtype=np.float32
        )
        self._future_masks: np.ndarray = np.ascontiguousarray(
            np.concatenate(fut_mask_chunks, axis=0), dtype=bool
        )
        self._future_offsets: np.ndarray = np.empty(len(fut_lens) + 1, dtype=np.int64)
        self._future_offsets[0] = 0
        np.cumsum(fut_lens, out=self._future_offsets[1:])

        self._future_target_positions: np.ndarray = np.ascontiguousarray(
            np.concatenate(target_pos_chunks, axis=0), dtype=np.int64
        )
        self._target_raw: np.ndarray = np.ascontiguousarray(
            np.concatenate(target_raw_chunks, axis=0), dtype=np.float32
        )
        self._target_mask: np.ndarray = np.ascontiguousarray(
            np.concatenate(target_mask_chunks, axis=0), dtype=bool
        )
        last_obs_raw = np.ascontiguousarray(
            np.concatenate(last_obs_list, axis=0), dtype=np.float32
        )

        # Target e last_obs nella scala di training (normalizzata o grezza)
        if self.preprocessing.normalize_targets:
            assert self.scaler is not None
            fill_val = np.float32(self.preprocessing.nan_fill_value)
            target_scaled = self.scaler.transform_targets(self._target_raw)
            self._target: np.ndarray = np.ascontiguousarray(
                np.where(self._target_mask, fill_val, target_scaled), dtype=np.float32
            )
            self._last_obs: np.ndarray = np.ascontiguousarray(
                self.scaler.transform_targets(last_obs_raw), dtype=np.float32
            )
        else:
            self._target = self._target_raw
            self._last_obs = last_obs_raw

        self._sample_indices: np.ndarray = np.ascontiguousarray(
            np.concatenate(sample_idx_chunks, axis=0), dtype=np.int64
        )
        self._areas: list[str] = area_list
        self._climates: list[str] = climate_list

    def __len__(self) -> int:
        return int(self._target_raw.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Indice campione {idx} fuori range [0, {len(self) - 1}].")

        h_start = int(self._history_offsets[idx])
        h_end = int(self._history_offsets[idx + 1])
        f_start = int(self._future_offsets[idx])
        f_end = int(self._future_offsets[idx + 1])

        return {
            "history": torch.from_numpy(self._history_values[h_start:h_end]),
            "history_mask": torch.from_numpy(self._history_masks[h_start:h_end]),
            "future": torch.from_numpy(self._future_values[f_start:f_end]),
            "future_mask": torch.from_numpy(self._future_masks[f_start:f_end]),
            "future_target_positions": torch.from_numpy(self._future_target_positions[idx]),
            "last_obs": torch.from_numpy(self._last_obs[idx]),
            "target": torch.from_numpy(self._target[idx]),
            "target_raw": torch.from_numpy(self._target_raw[idx]),
            "target_mask": torch.from_numpy(self._target_mask[idx]),
            "area": self._areas[idx],
            "climate": self._climates[idx],
            "sample_idx": int(self._sample_indices[idx]),
        }


# =============================================================================
# 4. Collate function con right-padding dinamico/fisso e feature delta_t
# =============================================================================


def collate_vegetation_batch(
    samples: Sequence[Mapping[str, Any]],
    *,
    dynamic_padding: bool = True,
    max_seq_len: int = 150,
    nan_fill_value: float = 0.0,
    add_delta_t: bool = True,
) -> Batch:
    """Aggrega una lista di campioni in un `Batch` conforme a `docs/batch_contract.md`.

    - Applica **right-padding** dinamico (fino a `max(history_len)` e `max(future_len)`
      nel batch) oppure fisso a `max_seq_len` se `dynamic_padding=False`.
    - Imposta `True` su tutte le posizioni di padding sia in `*_padding_mask` sia in
      `*_mask`, e riempie i valori float di padding con `nan_fill_value`.
    - Calcola in modo vettorizzato `history_delta_t`, `future_delta_t` e
      `target_delta_t` (o li azzera se `add_delta_t=False`).
    """
    batch_size = len(samples)
    if batch_size == 0:
        raise ValueError("Impossibile fare collate di un batch vuoto.")

    h_lens = [int(s["history"].shape[0]) for s in samples]
    f_lens = [int(s["future"].shape[0]) for s in samples]
    num_features = int(samples[0]["history"].shape[1])

    max_h_in_batch = max(h_lens)
    max_f_in_batch = max(f_lens)

    if dynamic_padding:
        t_h = max_h_in_batch
        t_f = max_f_in_batch
    else:
        if max_h_in_batch > max_seq_len:
            raise ValueError(
                f"Sequenza history di lunghezza {max_h_in_batch} supera max_seq_len={max_seq_len}."
            )
        if max_f_in_batch > max_seq_len:
            raise ValueError(
                f"Sequenza future di lunghezza {max_f_in_batch} supera max_seq_len={max_seq_len}."
            )
        t_h = max_seq_len
        t_f = max_seq_len

    history = torch.full((batch_size, t_h, num_features), nan_fill_value, dtype=torch.float32)
    history_mask = torch.ones((batch_size, t_h, num_features), dtype=torch.bool)
    history_padding_mask = torch.ones((batch_size, t_h), dtype=torch.bool)
    history_len = torch.tensor(h_lens, dtype=torch.int64)

    future = torch.full((batch_size, t_f, num_features), nan_fill_value, dtype=torch.float32)
    future_mask = torch.ones((batch_size, t_f, num_features), dtype=torch.bool)
    future_padding_mask = torch.ones((batch_size, t_f), dtype=torch.bool)
    future_len = torch.tensor(f_lens, dtype=torch.int64)

    for b, s in enumerate(samples):
        lh = h_lens[b]
        lf = f_lens[b]

        history[b, :lh] = s["history"]
        history_mask[b, :lh] = s["history_mask"]
        history_padding_mask[b, :lh] = False

        future[b, :lf] = s["future"]
        future_mask[b, :lf] = s["future_mask"]
        future_padding_mask[b, :lf] = False

    future_target_positions = torch.stack(
        [s["future_target_positions"] for s in samples], dim=0
    ).to(torch.int64)
    last_obs = torch.stack([s["last_obs"] for s in samples], dim=0).to(torch.float32)
    target = torch.stack([s["target"] for s in samples], dim=0).to(torch.float32)
    target_raw = torch.stack([s["target_raw"] for s in samples], dim=0).to(torch.float32)
    target_mask = torch.stack([s["target_mask"] for s in samples], dim=0).to(torch.bool)
    sample_idx = torch.tensor([int(s["sample_idx"]) for s in samples], dtype=torch.int64)

    if add_delta_t:
        h_steps = torch.arange(t_h, dtype=torch.int64).unsqueeze(0)  # (1, T_h)
        f_steps = torch.arange(t_f, dtype=torch.int64).unsqueeze(0)  # (1, T_f)
        zero = torch.zeros(1, dtype=torch.float32)
        history_delta_t = torch.where(
            history_padding_mask,
            zero,
            (h_steps - (history_len.unsqueeze(1) - 1)).to(torch.float32),
        )
        future_delta_t = torch.where(
            future_padding_mask,
            zero,
            (f_steps + 1).to(torch.float32),
        )
        target_delta_t = (future_target_positions + 1).to(torch.float32)
    else:
        history_delta_t = torch.zeros((batch_size, t_h), dtype=torch.float32)
        future_delta_t = torch.zeros((batch_size, t_f), dtype=torch.float32)
        target_delta_t = torch.zeros((batch_size, N_FUTURE), dtype=torch.float32)

    return {
        "history": history,
        "history_mask": history_mask,
        "history_padding_mask": history_padding_mask,
        "history_len": history_len,
        "history_delta_t": history_delta_t,
        "future": future,
        "future_mask": future_mask,
        "future_padding_mask": future_padding_mask,
        "future_len": future_len,
        "future_delta_t": future_delta_t,
        "future_target_positions": future_target_positions,
        "target_delta_t": target_delta_t,
        "last_obs": last_obs,
        "target": target,
        "target_raw": target_raw,
        "target_mask": target_mask,
        "area": [str(s["area"]) for s in samples],
        "climate": [str(s["climate"]) for s in samples],
        "sample_idx": sample_idx,
    }


class VegetationCollateFn:
    """Callable picklable per `DataLoader(..., collate_fn=...)` (compatibile con `num_workers > 0`)."""

    def __init__(
        self,
        *,
        dynamic_padding: bool = True,
        max_seq_len: int = 150,
        nan_fill_value: float = 0.0,
        add_delta_t: bool = True,
    ) -> None:
        self.dynamic_padding = bool(dynamic_padding)
        self.max_seq_len = int(max_seq_len)
        self.nan_fill_value = float(nan_fill_value)
        self.add_delta_t = bool(add_delta_t)

    @classmethod
    def from_preprocessing(cls, prep: PreprocessingConfig) -> "VegetationCollateFn":
        return cls(
            dynamic_padding=prep.dynamic_padding,
            max_seq_len=prep.max_seq_len,
            nan_fill_value=prep.nan_fill_value,
            add_delta_t=prep.add_delta_t,
        )

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> Batch:
        return collate_vegetation_batch(
            samples,
            dynamic_padding=self.dynamic_padding,
            max_seq_len=self.max_seq_len,
            nan_fill_value=self.nan_fill_value,
            add_delta_t=self.add_delta_t,
        )


# =============================================================================
# 5. Factory per Dataset e DataLoader da config.yaml
# =============================================================================


def build_dataset(
    split_name: str,
    cfg: Mapping[str, Any] | None = None,
    *,
    scaler: ArcsinhScaler | None = None,
    train_val_split: TrainValSplit | None = None,
    project_root: Path = PROJECT_ROOT,
    verify_json: bool = True,
) -> VegetationDataset:
    """Costruisce il `VegetationDataset` per uno split (`train`, `val`, `id`, `ood_s`, `ood_t`)."""
    if cfg is None:
        cfg = load_config()

    prep = PreprocessingConfig.from_config(cfg)
    if scaler is None and (prep.normalize_features or prep.normalize_targets):
        scaler = ArcsinhScaler.from_config(cfg, project_root=project_root)

    data_cfg = cfg["data"]
    data_root = Path(project_root) / data_cfg["root"]
    splits_cfg = data_cfg["splits"]

    if split_name in ("train", "val"):
        split_dir = data_root / splits_cfg["train"]
        if train_val_split is None:
            train_val_split = make_train_val_split(cfg, project_root=project_root)

        if train_val_split.mode == "area":
            areas = train_val_split.train_areas if split_name == "train" else train_val_split.val_areas
            return VegetationDataset(
                split_dir,
                scaler=scaler,
                preprocessing=prep,
                areas=areas,
                verify_json=verify_json,
            )
        if train_val_split.mode == "random":
            indices = (
                train_val_split.train_indices if split_name == "train" else train_val_split.val_indices
            )
            return VegetationDataset(
                split_dir,
                scaler=scaler,
                preprocessing=prep,
                indices=indices,
                verify_json=verify_json,
            )
        raise ValueError(f"Modalità TrainValSplit non riconosciuta: {train_val_split.mode}")

    if split_name not in splits_cfg:
        raise ValueError(
            f"Split '{split_name}' non riconosciuto. Valori validi: "
            f"['train', 'val', {', '.join(repr(k) for k in splits_cfg if k != 'train')}]."
        )

    split_dir = data_root / splits_cfg[split_name]
    return VegetationDataset(
        split_dir,
        scaler=scaler,
        preprocessing=prep,
        verify_json=verify_json,
    )


def build_dataloaders(
    cfg: Mapping[str, Any] | None = None,
    *,
    splits: Sequence[str] = ("train", "val", "id", "ood_s", "ood_t"),
    scaler: ArcsinhScaler | None = None,
    train_val_split: TrainValSplit | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    pin_memory: bool = False,
    project_root: Path = PROJECT_ROOT,
    verify_json: bool = True,
) -> dict[str, DataLoader[Batch]]:
    """Costruisce i `DataLoader` PyTorch per gli split richiesti leggendo `config.yaml`.

    - Lo scaler viene caricato una sola volta dal solo split di `train`.
    - Lo split `train`/`val` viene calcolato una sola volta e condiviso tra i due loader.
    - Solo `train` usa `shuffle=True` (con `torch.Generator` inizializzato da `cfg['seed']`).
    """
    if cfg is None:
        cfg = load_config()

    prep = PreprocessingConfig.from_config(cfg)
    if scaler is None and (prep.normalize_features or prep.normalize_targets):
        scaler = ArcsinhScaler.from_config(cfg, project_root=project_root)

    if train_val_split is None and any(s in ("train", "val") for s in splits):
        train_val_split = make_train_val_split(cfg, project_root=project_root)

    eff_batch_size = int(batch_size if batch_size is not None else cfg["training"]["batch_size"])
    eff_num_workers = int(num_workers if num_workers is not None else cfg["data"].get("num_workers", 0))
    seed = int(cfg.get("seed", 42))

    collate_fn = VegetationCollateFn.from_preprocessing(prep)
    loaders: dict[str, DataLoader[Batch]] = {}

    for split_name in splits:
        dataset = build_dataset(
            split_name,
            cfg,
            scaler=scaler,
            train_val_split=train_val_split,
            project_root=project_root,
            verify_json=verify_json,
        )
        is_train = split_name == "train"
        generator = torch.Generator().manual_seed(seed) if is_train else None
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=eff_batch_size,
            shuffle=is_train,
            num_workers=eff_num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=False,
            generator=generator,
        )

    return loaders
