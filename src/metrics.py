"""Metriche di valutazione per il forecasting probabilistico.

Le funzioni di questo modulo lavorano su array con il contratto del progetto:

* ``preds``: ``(B, H, C, Q)``;
* ``target``: ``(B, H, C)``;
* ``target_mask``: ``(B, H, C)``, con ``True`` = valore da escludere.

Le metriche non modificano gli input e ignorano sempre i target mascherati.
La funzione :func:`evaluate_predictions` restituisce sia il valore complessivo
sia le matrici disaggregate per orizzonte e indice, utili per tabelle e plot.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from src.contract import QUANTILES, TARGET_SHORT_NAMES


Array = np.ndarray


def _validate_quantiles(quantiles: Sequence[float]) -> Array:
    """Validate a strictly increasing quantile grid in ``[0, 1]``."""
    q = np.asarray(quantiles, dtype=np.float64)
    if q.ndim != 1 or q.size == 0:
        raise ValueError("quantiles deve essere un vettore non vuoto.")
    if not np.all(np.isfinite(q)) or np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("I quantili devono essere finiti e compresi tra 0 e 1.")
    if np.any(np.diff(q) <= 0.0):
        raise ValueError("quantiles deve essere strettamente crescente.")
    return q


def _validate_mask(target: Array, mask: Array | None) -> Array:
    """Normalize a target mask and check its shape."""
    if mask is None:
        return np.zeros_like(target, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != target.shape:
        raise ValueError(f"target_mask {mask.shape} incompatibile con target {target.shape}.")
    return mask


def inverse_standardization_arcsinh(
    values: Array,
    mean: Sequence[float] | float,
    var: Sequence[float] | float,
    eps: float = 1e-5,
) -> Array:
    """Riporta valori scalati con ``arcsinh((x - mean) / sqrt(var + eps))``."""
    values = np.asarray(values, dtype=np.float64)
    return np.sinh(values) * np.sqrt(np.asarray(var) + eps) + np.asarray(mean)


def inverse_transform_targets(values: Array, scaler: Mapping[str, Any]) -> Array:
    """Applica l'inversa dello scaler del train sull'ultimo asse target.

    ``values`` può avere shape ``(..., C)``. Sono accettati sia i nomi
    ``target_mean``/``target_var`` del file ``scaler.json`` sia ``mean``/``var``
    quando lo scaler contiene solo le statistiche dei target.
    """
    mean = scaler.get("target_mean", scaler.get("mean"))
    var = scaler.get("target_var", scaler.get("var"))
    if mean is None or var is None:
        raise ValueError("Lo scaler deve contenere target_mean/target_var oppure mean/var.")
    return inverse_standardization_arcsinh(values, mean, var, float(scaler.get("eps", 1e-5)))


def _validate_point_inputs(pred: Array, target: Array, mask: Array | None) -> tuple[Array, Array, Array]:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError(f"pred e target devono avere la stessa shape, ricevute {pred.shape} e {target.shape}.")
    if pred.ndim != 3:
        raise ValueError(f"Attesa shape (B, H, C), ricevuta {pred.shape}.")
    if mask is None:
        mask = np.zeros_like(target, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != target.shape:
            raise ValueError(f"target_mask {mask.shape} incompatibile con target {target.shape}.")
    valid = (~mask) & np.isfinite(pred) & np.isfinite(target)
    return pred, target, valid


def _masked_mean(values: Array, valid: Array) -> float:
    selected = np.asarray(values)[valid]
    return float(np.mean(selected)) if selected.size else float("nan")


def rmse(pred: Array, target: Array, target_mask: Array | None = None) -> float:
    pred, target, valid = _validate_point_inputs(pred, target, target_mask)
    return float(np.sqrt(_masked_mean((pred - target) ** 2, valid)))


def mae(pred: Array, target: Array, target_mask: Array | None = None) -> float:
    pred, target, valid = _validate_point_inputs(pred, target, target_mask)
    return _masked_mean(np.abs(pred - target), valid)


def wmape(pred: Array, target: Array, target_mask: Array | None = None, eps: float = 1e-8) -> float:
    """Weighted MAPE: ``sum(abs(error)) / sum(abs(target))``."""
    pred, target, valid = _validate_point_inputs(pred, target, target_mask)
    numerator = float(np.sum(np.abs(pred - target)[valid]))
    denominator = float(np.sum(np.abs(target)[valid]))
    return numerator / max(denominator, eps) if valid.any() else float("nan")


def mase(
    pred: Array,
    target: Array,
    scale: float | Array,
    target_mask: Array | None = None,
    eps: float = 1e-8,
) -> float:
    """Mean Absolute Scaled Error usando una scala calcolata sul solo train.

    ``scale`` deve essere il MAE dell'errore naïve scelto dal team (per esempio
    persistenza a un passo) e può essere scalare oppure per indice ``(C,)``.
    """
    pred, target, valid = _validate_point_inputs(pred, target, target_mask)
    scale_arr = np.asarray(scale, dtype=np.float64)
    if not np.all(np.isfinite(scale_arr)) or np.any(scale_arr < 0.0):
        raise ValueError("scale deve contenere valori finiti e non negativi.")
    if scale_arr.ndim == 1:
        if scale_arr.size != target.shape[-1]:
            raise ValueError("Una scala per indice deve avere lunghezza C.")
        scale_arr = scale_arr.reshape(1, 1, -1)
    return _masked_mean(np.abs(pred - target) / np.maximum(scale_arr, eps), valid)


def pinball(
    preds: Array,
    target: Array,
    quantiles: Sequence[float] = QUANTILES,
    target_mask: Array | None = None,
) -> float:
    """Pinball loss media su batch, orizzonti, indici e quantili."""
    preds = np.asarray(preds, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    q = _validate_quantiles(quantiles)
    if preds.shape != target.shape + (q.size,):
        raise ValueError(f"Attesa preds shape {target.shape + (q.size,)}, ricevuta {preds.shape}.")
    mask = _validate_mask(target, target_mask)
    error = target[..., None] - preds
    loss = np.maximum(q * error, (q - 1.0) * error)
    valid = (~mask)[..., None] & np.isfinite(loss)
    return _masked_mean(loss, valid)


def crps(
    preds: Array,
    target: Array,
    quantiles: Sequence[float] = QUANTILES,
    target_mask: Array | None = None,
) -> float:
    """Approximate the full CRPS from a finite quantile forecast.

    The task provides only a few quantiles. The implementation interpolates
    the quantile function between them and uses constant tails on ``[0, 1]``;
    therefore the result is an approximation, not an exact continuous CRPS.
    """
    """Approssimazione CRPS dalla previsione a quantili.

    Integra numericamente la pinball loss tra i quantili forniti. È una
    misura coerente per confrontare esperimenti con la stessa griglia
    ``(0.1, 0.5, 0.9)``; le code esterne non osservabili vengono omesse.
    """
    preds = np.asarray(preds, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    q = _validate_quantiles(quantiles)
    if q.size < 2:
        raise ValueError("CRPS richiede almeno due quantili.")
    if preds.shape != target.shape + (q.size,):
        raise ValueError("Shape incompatibile tra preds, target e quantiles.")
    mask = _validate_mask(target, target_mask)
    # Approximate the full [0, 1] CRPS using a piecewise-linear quantile
    # function and constant tails outside the supplied quantile grid.
    grid = np.unique(np.concatenate(([0.0], q, [1.0])))
    flat_preds = preds.reshape(-1, q.size)
    grid_preds = np.stack([np.interp(grid, q, row) for row in flat_preds], axis=0)
    grid_preds = grid_preds.reshape(target.shape + (grid.size,))
    error = target[..., None] - grid_preds
    point_pinball = np.maximum(grid * error, (grid - 1.0) * error)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    integrated = 2.0 * integrate(point_pinball, grid, axis=-1)
    return _masked_mean(integrated, (~mask) & np.isfinite(integrated))


def coverage(
    preds: Array,
    target: Array,
    target_mask: Array | None = None,
    lower_quantile: int = 0,
    upper_quantile: int = 2,
) -> float:
    """Copertura empirica dell'intervallo quantile, di default 80%."""
    preds = np.asarray(preds, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if preds.shape[:-1] != target.shape or preds.shape[-1] <= max(lower_quantile, upper_quantile):
        raise ValueError("Shape incompatibile tra preds e target.")
    mask = _validate_mask(target, target_mask)
    if lower_quantile < 0 or upper_quantile < 0:
        raise ValueError("Gli indici dei quantili devono essere non negativi.")
    if lower_quantile >= preds.shape[-1] or upper_quantile >= preds.shape[-1]:
        raise ValueError("Indice quantile fuori dai limiti.")
    if lower_quantile >= upper_quantile:
        raise ValueError("lower_quantile deve essere minore di upper_quantile.")
    valid = (~mask) & np.isfinite(target) & np.isfinite(preds[..., lower_quantile]) & np.isfinite(preds[..., upper_quantile])
    inside = (target >= preds[..., lower_quantile]) & (target <= preds[..., upper_quantile])
    return _masked_mean(inside.astype(np.float64), valid)


def _by_cell(metric_fn: Any, pred: Array, target: Array, mask: Array, **kwargs: Any) -> Array:
    """Calcola una metrica per ogni coppia (orizzonte, indice)."""
    result = np.full(target.shape[1:], np.nan, dtype=np.float64)
    for h in range(target.shape[1]):
        for c in range(target.shape[2]):
            cell_kwargs = dict(kwargs)
            if "scale" in cell_kwargs:
                scale = np.asarray(cell_kwargs["scale"])
                if scale.ndim == 1:
                    cell_kwargs["scale"] = scale[c]
            result[h, c] = metric_fn(pred[:, h : h + 1, c : c + 1], target[:, h : h + 1, c : c + 1], target_mask=mask[:, h : h + 1, c : c + 1], **cell_kwargs)
    return result


def evaluate_predictions(
    preds: Array,
    target_raw: Array,
    target_mask: Array | None = None,
    *,
    quantiles: Sequence[float] = QUANTILES,
    mase_scale: float | Array | None = None,
    group_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Genera il report completo per uno split o per un concatenato di split."""
    preds = np.asarray(preds, dtype=np.float64)
    target_raw = np.asarray(target_raw, dtype=np.float64)
    if preds.ndim != 4 or target_raw.shape != preds.shape[:-1]:
        raise ValueError("Attese preds (B,H,C,Q) e target_raw (B,H,C).")
    q = _validate_quantiles(quantiles)
    if not np.any(np.isclose(q, 0.5)):
        raise ValueError("quantiles deve contenere il quantile mediano 0.5.")
    mask = _validate_mask(target_raw, target_mask)
    median = preds[..., int(np.flatnonzero(np.isclose(q, 0.5))[0])]
    point: dict[str, Any] = {}
    for name, fn in (("rmse", rmse), ("mae", mae), ("wmape", wmape)):
        point[name] = {"overall": fn(median, target_raw, mask), "by_horizon_target": _by_cell(fn, median, target_raw, mask)}
    if mase_scale is not None:
        point["mase"] = {"overall": mase(median, target_raw, mase_scale, mask), "by_horizon_target": _by_cell(mase, median, target_raw, mask, scale=mase_scale)}
    uncertainty = {
        "pinball": pinball(preds, target_raw, q, mask),
        "crps": crps(preds, target_raw, q, mask),
        "coverage_80": coverage(preds, target_raw, mask),
    }
    uncertainty["pinball_by_horizon_target"] = _by_cell(lambda p, y, **kw: pinball(p, y, q, **kw), preds, target_raw, mask)
    uncertainty["crps_by_horizon_target"] = _by_cell(lambda p, y, **kw: crps(p, y, q, **kw), preds, target_raw, mask)
    uncertainty["coverage_80_by_horizon_target"] = _by_cell(coverage, preds, target_raw, mask)
    report = {"point": point, "uncertainty": uncertainty, "target_names": list(TARGET_SHORT_NAMES), "quantiles": list(q)}
    if group_ids is not None:
        report["by_group"] = evaluate_predictions_by_group(
            preds, target_raw, group_ids, target_mask=mask, quantiles=q, mase_scale=mase_scale
        )
    return report


def evaluate_predictions_by_group(
    preds: Array,
    target_raw: Array,
    group_ids: Sequence[str],
    target_mask: Array | None = None,
    *,
    quantiles: Sequence[float] = QUANTILES,
    mase_scale: float | Array | None = None,
) -> dict[str, Any]:
    """Return per-group reports plus an unweighted macro-average.

    The ordinary report is a micro-average over windows.  For this project,
    grouping by area prevents large areas from dominating spatial OOD results.
    """
    preds = np.asarray(preds, dtype=np.float64)
    target_raw = np.asarray(target_raw, dtype=np.float64)
    groups = np.asarray(group_ids)
    if groups.ndim != 1 or groups.size != preds.shape[0]:
        raise ValueError("group_ids deve avere una voce per ogni campione.")
    mask = _validate_mask(target_raw, target_mask)
    reports: dict[str, Any] = {}
    for group in np.unique(groups):
        idx = groups == group
        reports[str(group)] = evaluate_predictions(
            preds[idx], target_raw[idx], mask[idx], quantiles=quantiles, mase_scale=mase_scale
        )

    def macro(section: str, metric: str) -> float:
        values = []
        for group in reports:
            value = reports[group][section][metric]
            values.append(value["overall"] if isinstance(value, dict) else value)
        return float(np.nanmean(values)) if values else float("nan")

    macro_report: dict[str, Any] = {
        "point": {
            "rmse": macro("point", "rmse"),
            "mae": macro("point", "mae"),
            "wmape": macro("point", "wmape"),
        },
        "uncertainty": {
            "pinball": macro("uncertainty", "pinball"),
            "crps": macro("uncertainty", "crps"),
            "coverage_80": macro("uncertainty", "coverage_80"),
        },
        "n_groups": len(reports),
    }
    if mase_scale is not None:
        macro_report["point"]["mase"] = macro("point", "mase")
    return {"groups": reports, "macro": macro_report}

