"""Pipeline di preprocessing: normalizzazione `standardization_arcsinh` e split train/val.

Questo modulo implementa lo Step 3 della pipeline dati:
1. `ArcsinhScaler`: caricamento del solo `scaler.json` dello split di train,
   trasformazione `standardization_arcsinh` e trasformazione inversa per feature,
   target e predizioni `(B, 3, 3, 3)` (supporta sia `np.ndarray` sia `torch.Tensor`).
2. Split `train` / `val`:
   - `val_split_by: "area"` con opzione `val_balance_climate: true` per dividere
     le aree geografiche del train senza overlap, bilanciando sia la frazione
     complessiva di campioni (`val_fraction`) sia le classi climatiche di Köppen;
   - `val_split_by: "random"` per split casuale riproducibile a livello di campione.

Specifica del contratto del batch in `docs/batch_contract.md`.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar, overload

import numpy as np
import torch

from src.config import PROJECT_ROOT, load_config
from src.contract import N_DEFAULT_FEATURES, N_TARGETS

ArrayLike = TypeVar("ArrayLike", np.ndarray, torch.Tensor)

SUPPORTED_SCALER_MODE = "standardization_arcsinh"


# =============================================================================
# 1. Scaler standardization_arcsinh (solo da scaler.json del train)
# =============================================================================


class ArcsinhScaler:
    """Trasformazione `standardization_arcsinh` e inversa dalle statistiche del train.

    Formula diretta:
        z = (x - mean) / sqrt(var + eps)
        y = arcsinh(z)

    Formula inversa:
        x = sinh(y) * sqrt(var + eps) + mean

    I `NaN` presenti in ingresso vengono preservati (nessuna interpolazione).
    """

    def __init__(
        self,
        mean: Sequence[float] | np.ndarray,
        var: Sequence[float] | np.ndarray,
        target_mean: Sequence[float] | np.ndarray,
        target_var: Sequence[float] | np.ndarray,
        *,
        eps: float = 1e-5,
        mode: str = SUPPORTED_SCALER_MODE,
    ) -> None:
        if mode != SUPPORTED_SCALER_MODE:
            raise ValueError(
                f"Modalità scaler non supportata: '{mode}'. Attesa '{SUPPORTED_SCALER_MODE}'."
            )
        if eps <= 0.0:
            raise ValueError(f"eps deve essere strettamente positivo, trovato {eps}.")

        self.mode = mode
        self.eps = float(eps)

        self.mean = np.asarray(mean, dtype=np.float64)
        self.var = np.asarray(var, dtype=np.float64)
        self.target_mean = np.asarray(target_mean, dtype=np.float64)
        self.target_var = np.asarray(target_var, dtype=np.float64)

        if self.mean.ndim != 1 or self.var.ndim != 1 or self.mean.shape != self.var.shape:
            raise ValueError(
                f"mean {self.mean.shape} e var {self.var.shape} devono essere vettori 1D di uguale lunghezza."
            )
        if len(self.mean) != N_DEFAULT_FEATURES:
            raise ValueError(
                f"Attese {N_DEFAULT_FEATURES} feature nello scaler, trovate {len(self.mean)}."
            )
        if (
            self.target_mean.ndim != 1
            or self.target_var.ndim != 1
            or self.target_mean.shape != self.target_var.shape
        ):
            raise ValueError(
                f"target_mean {self.target_mean.shape} e target_var {self.target_var.shape} "
                "devono essere vettori 1D di uguale lunghezza."
            )
        if len(self.target_mean) != N_TARGETS:
            raise ValueError(
                f"Attesi {N_TARGETS} target nello scaler, trovati {len(self.target_mean)}."
            )
        if not (
            np.all(np.isfinite(self.mean))
            and np.all(np.isfinite(self.var))
            and np.all(np.isfinite(self.target_mean))
            and np.all(np.isfinite(self.target_var))
        ):
            raise ValueError("Lo scaler contiene valori non finiti (NaN o Inf).")
        if np.any(self.var < 0.0) or np.any(self.target_var < 0.0):
            raise ValueError("Le varianze nello scaler non possono essere negative.")

        self.std = np.sqrt(self.var + self.eps)
        self.target_std = np.sqrt(self.target_var + self.eps)

    @classmethod
    def from_json(cls, path: str | Path) -> "ArcsinhScaler":
        """Carica lo scaler da un file `scaler.json`."""
        scaler_path = Path(path)
        if not scaler_path.is_file():
            raise FileNotFoundError(f"File scaler non trovato: {scaler_path}")
        with open(scaler_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            mean=data["mean"],
            var=data["var"],
            target_mean=data["target_mean"],
            target_var=data["target_var"],
            eps=float(data.get("eps", 1e-5)),
            mode=str(data.get("mode", "")),
        )

    @classmethod
    def from_config(
        cls,
        cfg: Mapping[str, Any] | None = None,
        *,
        project_root: Path = PROJECT_ROOT,
    ) -> "ArcsinhScaler":
        """Carica lo scaler del train indicato in `config.yaml` (`data.scaler_file`)."""
        if cfg is None:
            cfg = load_config()
        data_cfg = cfg["data"]
        scaler_path = Path(project_root) / data_cfg["root"] / data_cfg["scaler_file"]
        return cls.from_json(scaler_path)

    # ------------------------------------------------------------------
    # Helper interni per NumPy e PyTorch
    # ------------------------------------------------------------------

    @staticmethod
    def _reshape_stats_np(
        stats: np.ndarray,
        ndim: int,
        axis: int,
        dtype: np.dtype,
    ) -> np.ndarray:
        norm_axis = axis % ndim
        shape = [1] * ndim
        shape[norm_axis] = stats.shape[0]
        return stats.astype(dtype, copy=False).reshape(shape)

    @staticmethod
    def _reshape_stats_torch(
        stats: np.ndarray,
        ref: torch.Tensor,
        axis: int,
    ) -> torch.Tensor:
        norm_axis = axis % ref.ndim
        shape = [1] * ref.ndim
        shape[norm_axis] = stats.shape[0]
        return torch.as_tensor(stats, dtype=ref.dtype, device=ref.device).reshape(shape)

    def _select_feature_stats(
        self,
        feature_indices: Sequence[int] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if feature_indices is None:
            return self.mean, self.std
        idx = np.asarray(feature_indices, dtype=np.int64)
        return self.mean[idx], self.std[idx]

    @overload
    def _apply_forward(
        self,
        x: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        axis: int,
    ) -> np.ndarray: ...

    @overload
    def _apply_forward(
        self,
        x: torch.Tensor,
        mean: np.ndarray,
        std: np.ndarray,
        axis: int,
    ) -> torch.Tensor: ...

    def _apply_forward(
        self,
        x: np.ndarray | torch.Tensor,
        mean: np.ndarray,
        std: np.ndarray,
        axis: int,
    ) -> np.ndarray | torch.Tensor:
        if isinstance(x, torch.Tensor):
            if not x.is_floating_point():
                x = x.to(torch.float32)
            norm_axis = axis % x.ndim
            if x.shape[norm_axis] != mean.shape[0]:
                raise ValueError(
                    f"Dimensione lungo asse {axis} ({x.shape[norm_axis]}) "
                    f"!= dimensione scaler ({mean.shape[0]})."
                )
            m = self._reshape_stats_torch(mean, x, norm_axis)
            s = self._reshape_stats_torch(std, x, norm_axis)
            return torch.arcsinh((x - m) / s)

        arr = np.asarray(x)
        if not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32)
        norm_axis = axis % arr.ndim
        if arr.shape[norm_axis] != mean.shape[0]:
            raise ValueError(
                f"Dimensione lungo asse {axis} ({arr.shape[norm_axis]}) "
                f"!= dimensione scaler ({mean.shape[0]})."
            )
        m = self._reshape_stats_np(mean, arr.ndim, norm_axis, arr.dtype)
        s = self._reshape_stats_np(std, arr.ndim, norm_axis, arr.dtype)
        return np.arcsinh((arr - m) / s)

    @overload
    def _apply_inverse(
        self,
        y: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        axis: int,
    ) -> np.ndarray: ...

    @overload
    def _apply_inverse(
        self,
        y: torch.Tensor,
        mean: np.ndarray,
        std: np.ndarray,
        axis: int,
    ) -> torch.Tensor: ...

    def _apply_inverse(
        self,
        y: np.ndarray | torch.Tensor,
        mean: np.ndarray,
        std: np.ndarray,
        axis: int,
    ) -> np.ndarray | torch.Tensor:
        if isinstance(y, torch.Tensor):
            if not y.is_floating_point():
                y = y.to(torch.float32)
            norm_axis = axis % y.ndim
            if y.shape[norm_axis] != mean.shape[0]:
                raise ValueError(
                    f"Dimensione lungo asse {axis} ({y.shape[norm_axis]}) "
                    f"!= dimensione scaler ({mean.shape[0]})."
                )
            m = self._reshape_stats_torch(mean, y, norm_axis)
            s = self._reshape_stats_torch(std, y, norm_axis)
            return torch.sinh(y) * s + m

        arr = np.asarray(y)
        if not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32)
        norm_axis = axis % arr.ndim
        if arr.shape[norm_axis] != mean.shape[0]:
            raise ValueError(
                f"Dimensione lungo asse {axis} ({arr.shape[norm_axis]}) "
                f"!= dimensione scaler ({mean.shape[0]})."
            )
        m = self._reshape_stats_np(mean, arr.ndim, norm_axis, arr.dtype)
        s = self._reshape_stats_np(std, arr.ndim, norm_axis, arr.dtype)
        return np.sinh(arr) * s + m

    # ------------------------------------------------------------------
    # API pubblica: feature, target, predizioni
    # ------------------------------------------------------------------

    def transform_features(
        self,
        x: ArrayLike,
        *,
        feature_indices: Sequence[int] | None = None,
        axis: int = -1,
    ) -> ArrayLike:
        """Applica `standardization_arcsinh` alle feature (`(..., F)`)."""
        mean, std = self._select_feature_stats(feature_indices)
        return self._apply_forward(x, mean, std, axis=axis)  # type: ignore[return-value]

    def inverse_transform_features(
        self,
        y: ArrayLike,
        *,
        feature_indices: Sequence[int] | None = None,
        axis: int = -1,
    ) -> ArrayLike:
        """Inverte `standardization_arcsinh` sulle feature (`(..., F)`)."""
        mean, std = self._select_feature_stats(feature_indices)
        return self._apply_inverse(y, mean, std, axis=axis)  # type: ignore[return-value]

    def transform_targets(
        self,
        y: ArrayLike,
        *,
        axis: int = -1,
    ) -> ArrayLike:
        """Applica `standardization_arcsinh` ai target o a `last_obs` (`(..., 3)`)."""
        return self._apply_forward(y, self.target_mean, self.target_std, axis=axis)  # type: ignore[return-value]

    def inverse_transform_targets(
        self,
        y: ArrayLike,
        *,
        axis: int = -1,
    ) -> ArrayLike:
        """Inverte `standardization_arcsinh` sui target (`(..., 3)`)."""
        return self._apply_inverse(y, self.target_mean, self.target_std, axis=axis)  # type: ignore[return-value]

    def inverse_transform_predictions(self, preds: ArrayLike) -> ArrayLike:
        """Inverte `standardization_arcsinh` sulle predizioni `(B, H=3, C=3, Q=3)`.

        L'asse degli indici vegetativi `C=3` nelle predizioni è `-2` (mentre `-1`
        è l'asse dei quantili `[0.1, 0.5, 0.9]`).
        """
        if preds.ndim < 2 or preds.shape[-2] != N_TARGETS:
            raise ValueError(
                f"Atteso tensore predizioni con asse -2 pari a {N_TARGETS} "
                f"(es. (B, 3, 3, 3)), trovata shape {tuple(preds.shape)}."
            )
        return self._apply_inverse(preds, self.target_mean, self.target_std, axis=-2)  # type: ignore[return-value]


# =============================================================================
# 2. Split Train / Validation (per area bilanciato su campioni e climi, o random)
# =============================================================================


@dataclass(frozen=True)
class AreaInfo:
    """Metadati aggregati di una singola area geografica nello split di train."""

    area: str
    n_samples: int
    climate_counts: dict[str, int]

    @property
    def dominant_climate(self) -> str:
        if not self.climate_counts:
            return "unknown"
        return max(self.climate_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


@dataclass(frozen=True)
class TrainValSplit:
    """Risultato dello split train/val, pronto per istanziare i Dataset."""

    mode: str  # "area" | "random"
    seed: int
    target_val_fraction: float
    balance_climate: bool
    train_areas: tuple[str, ...]
    val_areas: tuple[str, ...]
    n_train: int
    n_val: int
    train_climate_counts: dict[str, int]
    val_climate_counts: dict[str, int]
    # Indici globali (popolati solo quando mode == "random"; None quando mode == "area")
    train_indices: np.ndarray | None = None
    val_indices: np.ndarray | None = None

    @property
    def total_samples(self) -> int:
        return self.n_train + self.n_val

    @property
    def val_fraction_actual(self) -> float:
        return self.n_val / self.total_samples if self.total_samples > 0 else 0.0

    def summary(self) -> str:
        """Restituisce un riepilogo leggibile dello split e del bilanciamento climatico."""
        lines = [
            f"Train/Val split (mode={self.mode}, balance_climate={self.balance_climate}, seed={self.seed})",
            f"  Campioni: train={self.n_train:,} ({100 * (1 - self.val_fraction_actual):.2f}%), "
            f"val={self.n_val:,} ({100 * self.val_fraction_actual:.2f}%, target={100 * self.target_val_fraction:.1f}%)",
        ]
        if self.mode == "area":
            lines.append(
                f"  Aree:     train={len(self.train_areas)}, val={len(self.val_areas)} "
                f"(val_areas={list(self.val_areas)})"
            )
        all_climates = sorted(
            set(self.train_climate_counts) | set(self.val_climate_counts),
            key=lambda c: -(self.train_climate_counts.get(c, 0) + self.val_climate_counts.get(c, 0)),
        )
        lines.append("  Climi (train / val / %val):")
        for c in all_climates:
            tr = self.train_climate_counts.get(c, 0)
            va = self.val_climate_counts.get(c, 0)
            tot = tr + va
            pct = 100.0 * va / tot if tot > 0 else 0.0
            lines.append(f"    {c:<8s}: {tr:6d} / {va:6d} ({pct:5.2f}%)")
        return "\n".join(lines)


def collect_area_infos(split_dir: str | Path) -> list[AreaInfo]:
    """Legge i file `<area>.json` di uno split e raccoglie conteggi per area e clima."""
    split_path = Path(split_dir)
    if not split_path.is_dir():
        raise FileNotFoundError(f"Cartella split non trovata: {split_path}")

    json_files = sorted(p for p in split_path.glob("*.json") if p.name != "scaler.json")
    if not json_files:
        raise ValueError(f"Nessun file metadati <area>.json trovato in {split_path}")

    infos: list[AreaInfo] = []
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            meta = json.load(f)
        records = meta.get("metadata_records", [])
        clim_counts = dict(Counter(str(r.get("climate", "unknown")) for r in records))
        infos.append(
            AreaInfo(
                area=jf.stem,
                n_samples=len(records),
                climate_counts=clim_counts,
            )
        )
    return infos


def split_areas_balanced(
    area_infos: Sequence[AreaInfo],
    *,
    val_fraction: float = 0.15,
    balance_climate: bool = True,
    seed: int = 42,
    n_restarts: int = 24,
) -> TrainValSplit:
    """Divide le aree geografiche in `train` e `val` senza overlap.

    Quando `balance_climate=True`, ottimizza congiuntamente:
    1. aderenza della frazione complessiva di campioni a `val_fraction`;
    2. aderenza della frazione di validazione per ciascuna classe di Köppen
       (sia pesata per volume della classe sia uniforme sulle classi presenti
       in almeno 2 aree);
    3. vincolo rigido: nessuna classe climatica presente nel dataset viene mai
       azzerata nel `train` (es. classi con una sola area restano sempre in `train`).

    L'ottimizzazione usa `n_restarts` inizializzazioni greedy randomizzate (con
    `seed` deterministico) seguite da ricerca locale 1-opt (flip) e 2-opt (swap).
    """
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction deve essere in (0, 1), trovato {val_fraction}.")
    if len(area_infos) < 2:
        raise ValueError("Servono almeno 2 aree per eseguire uno split per area.")

    areas = sorted(area_infos, key=lambda a: a.area)
    n_areas = len(areas)

    # Costruisce matrice (n_areas, n_climates) dei conteggi climatici
    climates = sorted({c for a in areas for c in a.climate_counts})
    clim_to_idx = {c: i for i, c in enumerate(climates)}
    n_clim = len(climates)

    sample_counts = np.array([a.n_samples for a in areas], dtype=np.int64)
    total_samples = int(sample_counts.sum())
    if total_samples == 0:
        raise ValueError("Totale campioni pari a 0.")

    clim_matrix = np.zeros((n_areas, n_clim), dtype=np.int64)
    for i, a in enumerate(areas):
        for c, cnt in a.climate_counts.items():
            clim_matrix[i, clim_to_idx[c]] = cnt

    total_clim = clim_matrix.sum(axis=0)  # (n_clim,)
    areas_per_clim = (clim_matrix > 0).sum(axis=0)  # (n_clim,)
    multi_area_clim_mask = areas_per_clim >= 2
    clim_weights = total_clim / float(total_samples)
    n_multi_clim = int(multi_area_clim_mask.sum())

    # Aree bloccate in train: contengono l'unica occorrenza di una classe climatica
    locked_train = np.zeros(n_areas, dtype=bool)
    if balance_climate:
        for k in range(n_clim):
            if areas_per_clim[k] == 1:
                locked_train |= clim_matrix[:, k] > 0

    def is_valid_state(val_mask: np.ndarray, val_clim: np.ndarray) -> bool:
        n_val_areas = int(val_mask.sum())
        if n_val_areas == 0 or n_val_areas == n_areas:
            return False
        if balance_climate:
            # Nessun clima può scomparire dal train
            if np.any((total_clim - val_clim) <= 0):
                return False
        return True

    def objective(val_samples: int, val_clim: np.ndarray) -> float:
        total_err = (val_samples / float(total_samples) - val_fraction) ** 2
        if not balance_climate or n_clim == 0:
            return total_err

        clim_fracs = val_clim / np.maximum(total_clim, 1)
        clim_sq_err = (clim_fracs - val_fraction) ** 2
        weighted_clim_err = float(np.sum(clim_weights * clim_sq_err))
        if n_multi_clim > 0:
            uniform_clim_err = float(np.mean(clim_sq_err[multi_area_clim_mask]))
        else:
            uniform_clim_err = 0.0
        # Pesi calibrati: forte aderenza al target globale + bilanciamento classi principali e minori
        return 10.0 * total_err + 4.0 * weighted_clim_err + 1.0 * uniform_clim_err

    rng = np.random.default_rng(seed)
    eligible = np.where(~locked_train)[0]
    if len(eligible) == 0:
        raise ValueError("Tutte le aree sono bloccate in train: impossibile creare uno split.")

    best_mask: np.ndarray | None = None
    best_score = float("inf")

    for restart in range(max(1, n_restarts)):
        val_mask = np.zeros(n_areas, dtype=bool)
        val_samples = 0
        val_clim = np.zeros(n_clim, dtype=np.int64)

        # Inizializzazione greedy: primo restart ordinato per dimensione decrescente,
        # successivi con permutazione casuale controllata dal seed
        if restart == 0:
            order = eligible[np.argsort(-sample_counts[eligible], kind="mergesort")]
        else:
            order = rng.permutation(eligible)

        cur_score = objective(val_samples, val_clim)
        for idx in order:
            new_samples = val_samples + int(sample_counts[idx])
            new_clim = val_clim + clim_matrix[idx]
            if balance_climate and np.any((total_clim - new_clim) <= 0):
                continue
            new_score = objective(new_samples, new_clim)
            if new_score < cur_score or int(val_mask.sum()) == 0:
                val_mask[idx] = True
                val_samples = new_samples
                val_clim = new_clim
                cur_score = new_score

        # Ricerca locale 1-opt (flip singolo) e 2-opt (swap train <-> val)
        improved = True
        while improved:
            improved = False

            # 1-opt: aggiungi o rimuovi una singola area
            best_flip = -1
            best_flip_score = cur_score - 1e-12
            for idx in eligible:
                if val_mask[idx]:
                    if int(val_mask.sum()) <= 1:
                        continue
                    cand_samples = val_samples - int(sample_counts[idx])
                    cand_clim = val_clim - clim_matrix[idx]
                else:
                    if int(val_mask.sum()) >= n_areas - 1:
                        continue
                    cand_samples = val_samples + int(sample_counts[idx])
                    cand_clim = val_clim + clim_matrix[idx]
                    if balance_climate and np.any((total_clim - cand_clim) <= 0):
                        continue
                s = objective(cand_samples, cand_clim)
                if s < best_flip_score:
                    best_flip_score = s
                    best_flip = int(idx)

            if best_flip != -1:
                if val_mask[best_flip]:
                    val_mask[best_flip] = False
                    val_samples -= int(sample_counts[best_flip])
                    val_clim -= clim_matrix[best_flip]
                else:
                    val_mask[best_flip] = True
                    val_samples += int(sample_counts[best_flip])
                    val_clim += clim_matrix[best_flip]
                cur_score = best_flip_score
                improved = True
                continue

            # 2-opt: scambia un'area in train con una in val
            val_indices = np.where(val_mask)[0]
            train_indices = np.where((~val_mask) & (~locked_train))[0]
            best_swap: tuple[int, int] | None = None
            best_swap_score = cur_score - 1e-12

            for i_val in val_indices:
                base_samples = val_samples - int(sample_counts[i_val])
                base_clim = val_clim - clim_matrix[i_val]
                for i_tr in train_indices:
                    cand_samples = base_samples + int(sample_counts[i_tr])
                    cand_clim = base_clim + clim_matrix[i_tr]
                    if balance_climate and np.any((total_clim - cand_clim) <= 0):
                        continue
                    s = objective(cand_samples, cand_clim)
                    if s < best_swap_score:
                        best_swap_score = s
                        best_swap = (int(i_val), int(i_tr))

            if best_swap is not None:
                i_val, i_tr = best_swap
                val_mask[i_val] = False
                val_mask[i_tr] = True
                val_samples = val_samples - int(sample_counts[i_val]) + int(sample_counts[i_tr])
                val_clim = val_clim - clim_matrix[i_val] + clim_matrix[i_tr]
                cur_score = best_swap_score
                improved = True

        if is_valid_state(val_mask, val_clim) and cur_score < best_score:
            best_score = cur_score
            best_mask = val_mask.copy()

    if best_mask is None:
        raise RuntimeError("Impossibile trovare uno split train/val valido per le aree fornite.")

    train_areas = tuple(areas[i].area for i in range(n_areas) if not best_mask[i])
    val_areas = tuple(areas[i].area for i in range(n_areas) if best_mask[i])

    train_clim_vec = clim_matrix[~best_mask].sum(axis=0)
    val_clim_vec = clim_matrix[best_mask].sum(axis=0)

    train_climate_counts = {
        climates[k]: int(train_clim_vec[k]) for k in range(n_clim) if train_clim_vec[k] > 0
    }
    val_climate_counts = {
        climates[k]: int(val_clim_vec[k]) for k in range(n_clim) if val_clim_vec[k] > 0
    }

    return TrainValSplit(
        mode="area",
        seed=seed,
        target_val_fraction=val_fraction,
        balance_climate=balance_climate,
        train_areas=train_areas,
        val_areas=val_areas,
        n_train=int(sample_counts[~best_mask].sum()),
        n_val=int(sample_counts[best_mask].sum()),
        train_climate_counts=train_climate_counts,
        val_climate_counts=val_climate_counts,
    )


def split_samples_random(
    area_infos: Sequence[AreaInfo],
    *,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> TrainValSplit:
    """Split casuale deterministico a livello di campione (`val_split_by: "random"`)."""
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction deve essere in (0, 1), trovato {val_fraction}.")

    areas = sorted(area_infos, key=lambda a: a.area)
    total_samples = sum(a.n_samples for a in areas)
    if total_samples < 2:
        raise ValueError("Servono almeno 2 campioni per eseguire uno split random.")

    n_val = max(1, min(total_samples - 1, int(round(total_samples * val_fraction))))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total_samples)
    val_indices = np.sort(perm[:n_val])
    train_indices = np.sort(perm[n_val:])

    # Stima esatta dei climi se ciascuna area ha un solo clima, altrimenti conteggio totale
    all_areas = tuple(a.area for a in areas)
    sample_climates: list[str] = []
    for a in areas:
        if len(a.climate_counts) == 1:
            c = next(iter(a.climate_counts))
            sample_climates.extend([c] * a.n_samples)
        else:
            for c, cnt in sorted(a.climate_counts.items()):
                sample_climates.extend([c] * cnt)

    train_clim = dict(Counter(sample_climates[i] for i in train_indices))
    val_clim = dict(Counter(sample_climates[i] for i in val_indices))

    return TrainValSplit(
        mode="random",
        seed=seed,
        target_val_fraction=val_fraction,
        balance_climate=False,
        train_areas=all_areas,
        val_areas=all_areas,
        n_train=len(train_indices),
        n_val=len(val_indices),
        train_climate_counts=train_clim,
        val_climate_counts=val_clim,
        train_indices=train_indices,
        val_indices=val_indices,
    )


def make_train_val_split(
    cfg: Mapping[str, Any] | None = None,
    *,
    area_infos: Sequence[AreaInfo] | None = None,
    project_root: Path = PROJECT_ROOT,
) -> TrainValSplit:
    """Costruisce lo split train/val leggendo i parametri da `config.yaml`."""
    if cfg is None:
        cfg = load_config()

    data_cfg = cfg["data"]
    seed = int(cfg.get("seed", 42))
    val_fraction = float(data_cfg.get("val_fraction", 0.15))
    val_split_by = str(data_cfg.get("val_split_by", "area")).lower()
    val_balance_climate = bool(data_cfg.get("val_balance_climate", True))

    if area_infos is None:
        train_dir = Path(project_root) / data_cfg["root"] / data_cfg["splits"]["train"]
        area_infos = collect_area_infos(train_dir)

    if val_split_by == "area":
        return split_areas_balanced(
            area_infos,
            val_fraction=val_fraction,
            balance_climate=val_balance_climate,
            seed=seed,
        )
    if val_split_by == "random":
        return split_samples_random(
            area_infos,
            val_fraction=val_fraction,
            seed=seed,
        )
    raise ValueError(
        f"val_split_by non riconosciuto: '{val_split_by}'. Valori ammessi: 'area', 'random'."
    )
