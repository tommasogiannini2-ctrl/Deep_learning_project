"""Registro dei modelli e factory `build_model`.

Tutto il resto del progetto (training, valutazione) crea i modelli solo da qui:

    from src.models import build_model
    model = build_model(cfg)            # legge model.name e model.params da config.yaml
    preds = model(batch)                # (B, 3, 3, 3)

Per aggiungere un nuovo modello basta implementarlo in src/models/ e
registrarlo in MODEL_REGISTRY: training ed eval non vanno modificati.
"""

from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from src.contract import FEATURE_NAMES
from src.models.baselines import PersistenceModel, ResidualMLPBaseline

# Nome usato in config.yaml (model.name) -> classe del modello.
MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "persistence": PersistenceModel,
    "residual_mlp": ResidualMLPBaseline,
}


def infer_num_features(cfg: Mapping[str, Any]) -> int:
    """Numero di feature F che il modello riceverà, dato `preprocessing.drop_features`.

    Di default F = 21; ogni feature rimossa per ablation lo riduce di 1.
    Un nome sconosciuto produce un errore (probabile refuso in config.yaml).
    """
    drop = list((cfg.get("preprocessing") or {}).get("drop_features") or [])
    unknown = [name for name in drop if name not in FEATURE_NAMES]
    if unknown:
        raise ValueError(f"drop_features contiene feature sconosciute: {unknown}.")
    return len(FEATURE_NAMES) - len(set(drop))


def build_model(cfg: Mapping[str, Any], num_features: int | None = None) -> nn.Module:
    """Crea il modello indicato in config.yaml.

    Legge `model.name` (chiave di MODEL_REGISTRY) e i parametri del blocco
    `model.params.<name>`: i blocchi degli altri modelli vengono ignorati, così
    per cambiare modello basta cambiare `model.name`.

    Args:
        cfg: configurazione letta con load_config().
        num_features: numero di feature del batch. Se None viene ricavato da
            `preprocessing.drop_features` con infer_num_features().
    """
    model_cfg = cfg.get("model") or {}
    name = model_cfg.get("name")
    if not name:
        raise ValueError(f"model.name non impostato in config.yaml. Disponibili: {sorted(MODEL_REGISTRY)}.")
    name = str(name).lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Modello sconosciuto: {name!r}. Disponibili: {sorted(MODEL_REGISTRY)}.")

    if num_features is None:
        num_features = infer_num_features(cfg)

    params = dict((model_cfg.get("params") or {}).get(name) or {})
    try:
        return MODEL_REGISTRY[name](num_features=num_features, **params)
    except TypeError as exc:
        # Tipicamente un parametro scritto male in config.yaml.
        raise ValueError(f"Parametri non validi per il modello {name!r}: {exc}") from exc


__all__ = ["MODEL_REGISTRY", "PersistenceModel", "ResidualMLPBaseline", "build_model", "infer_num_features"]
