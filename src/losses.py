"""Funzioni di perdita probabilistiche per la previsione multi-quantile.

Il modello non prevede un solo valore per ogni indice e step futuro, ma tre
quantili (0.1, 0.5, 0.9). Per addestrarlo serve una loss che premi insieme:

- la calibrazione: il valore vero deve cadere sotto q0.1 circa nel 10% dei casi
  e sotto q0.9 circa nel 90% dei casi;
- la precisione: a parità di calibrazione, gli intervalli devono essere stretti.

La Pinball Loss (o quantile loss) fa esattamente questo. Per un quantile tau:

    L_tau(y, q) = max( tau * (y - q), (tau - 1) * (y - q) )

È asimmetrica: con tau = 0.9 stare sotto al valore vero costa 9 volte più che
stare sopra, quindi la rete impara a spingere q0.9 verso l'alto finché solo il
10% dei valori veri lo supera. Con tau = 0.5 è simmetrica (metà del MAE) e
porta alla mediana.

Convenzioni di forma (vedi docs/batch_contract.md):
    preds        (B, H, C, Q) = (batch, step futuro, indice, quantile)
    target       (B, H, C)
    target_mask  (B, H, C)    bool, True = target mancante (escluso dalla loss)

Uso tipico nel training loop:

    criterion = build_loss(cfg)
    loss = criterion(preds, batch["target"], batch["target_mask"])
    loss.backward()
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn

from src.contract import QUANTILES

# Modalità di riduzione supportate dalla loss.
_REDUCTIONS = ("mean", "sum", "none")


def _check_quantiles(quantiles: Sequence[float]) -> None:
    """Controlla che i quantili siano strettamente compresi tra 0 e 1 e crescenti."""
    q = list(quantiles)
    if not q:
        raise ValueError("La lista dei quantili è vuota.")
    if any(not 0.0 < x < 1.0 for x in q):
        raise ValueError(f"I quantili devono essere in (0, 1), trovati {q}.")
    if any(b <= a for a, b in zip(q, q[1:])):
        raise ValueError(f"I quantili devono essere strettamente crescenti, trovati {q}.")


def pinball_loss(
    preds: torch.Tensor,
    target: torch.Tensor,
    quantiles: torch.Tensor | Sequence[float] = QUANTILES,
    target_mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Pinball Loss sui quantili, con esclusione dei target mancanti.

    Args:
        preds: predizioni (B, H, C, Q); l'ultimo asse segue l'ordine di `quantiles`.
        target: valori veri (B, H, C), nella stessa scala di `preds`.
        quantiles: livelli dei quantili, lunghezza Q (default 0.1, 0.5, 0.9).
        target_mask: (B, H, C) bool, True = target mancante. Se None, tutti validi.
        reduction: "mean" = media sugli elementi validi (default, usata in training);
            "sum" = somma sugli elementi validi;
            "none" = tensore (B, H, C, Q) con zeri sulle posizioni mascherate.

    Returns:
        Uno scalare per "mean"/"sum", un tensore (B, H, C, Q) per "none".
    """
    if reduction not in _REDUCTIONS:
        raise ValueError(f"reduction deve essere uno tra {_REDUCTIONS}, trovato {reduction!r}.")

    # --- Controlli di forma: un errore chiaro qui evita broadcasting silenziosi sbagliati.
    if preds.ndim != 4:
        raise ValueError(f"preds deve avere 4 dimensioni (B, H, C, Q), trovato {tuple(preds.shape)}.")
    if tuple(target.shape) != tuple(preds.shape[:-1]):
        raise ValueError(
            f"target {tuple(target.shape)} incompatibile con preds {tuple(preds.shape)}: "
            "atteso target con shape preds.shape[:-1]."
        )
    if target_mask is not None and tuple(target_mask.shape) != tuple(target.shape):
        raise ValueError(
            f"target_mask {tuple(target_mask.shape)} deve avere la stessa shape di target {tuple(target.shape)}."
        )

    # --- Quantili come tensore sullo stesso device/dtype delle predizioni, shape (Q,).
    if not isinstance(quantiles, torch.Tensor):
        _check_quantiles(quantiles)
        quantiles = torch.tensor(list(quantiles))
    quantiles = quantiles.to(device=preds.device, dtype=preds.dtype)
    if quantiles.numel() != preds.shape[-1]:
        raise ValueError(
            f"Numero di quantili ({quantiles.numel()}) diverso dall'ultima dimensione di preds ({preds.shape[-1]})."
        )

    # --- Errore y - q per ogni quantile: target (B, H, C) -> (B, H, C, 1) per il broadcasting.
    errors = target.unsqueeze(-1) - preds  # (B, H, C, Q)

    # --- Esclusione dei target mancanti.
    # Azzeriamo l'errore PRIMA di calcolare la loss (invece di filtrare dopo):
    # così anche un eventuale NaN in un target mascherato non produce NaN né
    # gradienti spuri, e le posizioni mascherate ricevono gradiente esattamente 0.
    if target_mask is not None:
        valid = (~target_mask).unsqueeze(-1).expand_as(errors)  # (B, H, C, Q)
        errors = torch.where(valid, errors, torch.zeros_like(errors))
    else:
        valid = None

    # --- Pinball: tau * e se e >= 0 (sottostima), (tau - 1) * e se e < 0 (sovrastima).
    loss = torch.maximum(quantiles * errors, (quantiles - 1.0) * errors)  # (B, H, C, Q)

    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()

    # reduction == "mean": media solo sugli elementi validi.
    n_valid = loss.numel() if valid is None else int(valid.sum().item())
    if n_valid == 0:
        # Nessun target valido nel batch: loss nulla ma ancora collegata al grafo,
        # così backward() non fallisce e non introduce NaN (0 / 0).
        return loss.sum() * 0.0
    return loss.sum() / n_valid


class PinballLoss(nn.Module):
    """Versione nn.Module della Pinball Loss, da usare come criterio nel training.

    I quantili sono salvati come buffer: si spostano automaticamente su GPU con
    `criterion.to(device)` e non sono parametri addestrabili.
    """

    def __init__(self, quantiles: Sequence[float] = QUANTILES, reduction: str = "mean") -> None:
        super().__init__()
        _check_quantiles(quantiles)
        if reduction not in _REDUCTIONS:
            raise ValueError(f"reduction deve essere uno tra {_REDUCTIONS}, trovato {reduction!r}.")
        self.reduction = reduction
        self.register_buffer("quantiles", torch.tensor(list(quantiles), dtype=torch.float32))

    def forward(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return pinball_loss(preds, target, self.quantiles, target_mask, self.reduction)


def build_loss(cfg: Mapping[str, Any] | None = None) -> nn.Module:
    """Crea la loss indicata in config.yaml (chiave `training.loss`).

    I quantili vengono letti da `task.quantiles`. Per aggiungere una nuova loss
    (ad esempio per un'ablation) basta implementarla in questo file e
    aggiungere un ramo qui sotto: il training loop non va modificato.
    """
    cfg = cfg or {}
    name = str(cfg.get("training", {}).get("loss", "pinball")).lower()
    quantiles = cfg.get("task", {}).get("quantiles", QUANTILES)

    if name == "pinball":
        return PinballLoss(quantiles=quantiles)
    raise ValueError(f"Loss non supportata: {name!r}. Disponibili: 'pinball'.")
