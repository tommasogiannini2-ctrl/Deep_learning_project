"""Ciclo di addestramento, validazione e salvataggio dei checkpoint.

Il training è disaccoppiato dai dati: `train()` accetta qualsiasi coppia di
loader che restituiscano batch conformi al contratto (docs/batch_contract.md).
Quindi funziona sia con i DataLoader veri (src/dataset.py) sia con batch
sintetici (`make_dummy_loaders`), senza cambiare una riga del ciclo.

Cosa fa `train()`:
1. fissa i seed, sceglie il device e costruisce modello e loss da config.yaml;
2. per ogni epoca: addestra sul train, misura la loss sulla validazione,
   aggiorna il learning rate, salva il modello se la validazione migliora;
3. si ferma in anticipo se la validazione non migliora per `patience` epoche
   (early stopping) e restituisce il modello con i pesi dell'epoca migliore;
4. scrive in `runs/<experiment_name>/` tre file:
     best.pt           pesi migliori + configurazione usata (vedi load_model_from_checkpoint)
     config_used.yaml  copia leggibile della configurazione
     history.json      loss di train e validazione e learning rate per epoca

Per usare un modello già addestrato (ad esempio nella valutazione):

    model, ckpt = load_model_from_checkpoint("runs/default/best.pt")
    preds = model(batch)    # (B, 3, 3, 3), quantili ordinati
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml
from torch import nn

from src.config import PROJECT_ROOT
from src.contract import make_dummy_batch, move_batch_to_device, validate_batch
from src.losses import build_loss
from src.models import build_model, infer_num_features

# Versione del formato del checkpoint: da incrementare se cambia la struttura di best.pt.
CHECKPOINT_FORMAT_VERSION = 1

CHECKPOINT_NAME = "best.pt"
CONFIG_USED_NAME = "config_used.yaml"
HISTORY_NAME = "history.json"


# =============================================================================
# 1. Riproducibilità e device
# =============================================================================


def set_seed(seed: int) -> None:
    """Fissa i generatori casuali di Python, NumPy e PyTorch.

    Con lo stesso seed (e gli stessi dati) due addestramenti producono le stesse
    loss: inizializzazione dei pesi e dropout dipendono dal generatore di PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    """Traduce `training.device` di config.yaml: "auto" usa la GPU se disponibile, altrimenti la CPU."""
    name = str(name).lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training.device = 'cuda' ma nessuna GPU CUDA è disponibile: usa 'auto' o 'cpu'.")
    if name not in ("cpu", "cuda"):
        raise ValueError(f"training.device deve essere 'auto', 'cpu' o 'cuda', trovato {name!r}.")
    return torch.device(name)


# =============================================================================
# 2. Optimizer e scheduler
# =============================================================================


def build_optimizer(params: Iterable[nn.Parameter], tcfg: Mapping[str, Any]) -> torch.optim.Optimizer:
    """Crea l'optimizer indicato in `training.optimizer` (adam | adamw | sgd).

    - adamw (default): Adam con weight decay applicato direttamente ai pesi, separato
      dalle medie mobili dei gradienti: regolarizza meglio di Adam;
    - adam: come adamw ma il weight decay si mescola ai gradienti;
    - sgd: discesa del gradiente con momentum 0.9, più lento da tarare su serie temporali.
    """
    name = str(tcfg.get("optimizer", "adamw")).lower()
    lr = float(tcfg["learning_rate"])
    weight_decay = float(tcfg.get("weight_decay", 0.0))

    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"training.optimizer non supportato: {name!r}. Disponibili: adam, adamw, sgd.")


def build_scheduler(
    optimizer: torch.optim.Optimizer, tcfg: Mapping[str, Any], epochs: int
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    """Crea lo scheduler del learning rate indicato in `training.scheduler`.

    - cosine (default): il learning rate scende con una curva a coseno da lr a 0
      lungo tutte le `epochs`: passi grandi all'inizio, piccoli per rifinire;
    - plateau: dimezza il learning rate (`plateau_factor`) quando la loss di
      validazione non migliora per `plateau_patience` epoche;
    - none: learning rate costante.
    """
    name = str(tcfg.get("scheduler", "cosine")).lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(tcfg.get("plateau_factor", 0.5)),
            patience=int(tcfg.get("plateau_patience", 3)),
        )
    if name == "none":
        return None
    raise ValueError(f"training.scheduler non supportato: {name!r}. Disponibili: cosine, plateau, none.")


# =============================================================================
# 3. Una epoca di addestramento o valutazione
# =============================================================================


def run_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float | None = None,
    check_contract: bool = False,
) -> float:
    """Percorre una volta il loader e restituisce la loss media.

    Con `optimizer` aggiorna i pesi (training); senza, calcola solo la loss
    (validazione, modello in modalità eval e senza gradienti: il dropout è spento).

    La media è pesata per il numero di target validi di ogni batch: l'ultimo
    batch, più piccolo, conta quanto gli altri per elemento, quindi il risultato
    non dipende da come i campioni sono divisi in batch.

    Args:
        grad_clip: se indicato, limita la norma globale del gradiente a questo
            valore prima di ogni aggiornamento (protegge da picchi su batch estremi).
        check_contract: se True controlla il primo batch con `validate_batch`,
            per scoprire subito dati non conformi invece di avere errori oscuri.
    """
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_valid = 0
    for i, batch in enumerate(loader):
        if check_contract and i == 0:
            validate_batch(batch)
        batch = move_batch_to_device(batch, device)
        n_valid = int((~batch["target_mask"]).sum().item())

        with torch.set_grad_enabled(training):
            preds = model(batch)
            loss = criterion(preds, batch["target"], batch["target_mask"])

        # Una loss non finita significa dati o learning rate fuori controllo:
        # meglio fermarsi subito che proseguire con pesi corrotti.
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Loss non finita ({loss.item()}) al batch {i}: controlla i dati, il learning rate e il gradient clipping."
            )

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * n_valid
        total_valid += n_valid

    if total_valid == 0:
        raise ValueError("Il loader non contiene target validi (vuoto o tutti mascherati).")
    return total_loss / total_valid


# =============================================================================
# 4. Checkpoint
# =============================================================================


def save_checkpoint(
    path: str | Path,
    model_state: Mapping[str, torch.Tensor],
    cfg: Mapping[str, Any],
    num_features: int,
    best_val_loss: float,
    best_epoch: int,
) -> None:
    """Salva best.pt: pesi e tutto ciò che serve per ricreare il modello.

    Salvare anche la configurazione e `num_features` permette di ricaricare il
    modello in qualsiasi momento, anche dopo aver cambiato config.yaml (altrimenti
    cambiare ad esempio `hidden_dim` renderebbe illeggibili i vecchi pesi).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": dict(model_state),
            "config": copy.deepcopy(dict(cfg)),
            "num_features": int(num_features),
            "best_val_loss": float(best_val_loss),
            "best_epoch": int(best_epoch),
            "seed": int(cfg.get("seed", 42)),
        },
        path,
    )


def load_model_from_checkpoint(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[nn.Module, dict[str, Any]]:
    """Ricrea il modello salvato in un checkpoint, già in modalità eval.

    Usa la configurazione salvata nel checkpoint (non quella attuale di
    config.yaml), quindi l'architettura corrisponde sempre ai pesi.

    Returns:
        (model, ckpt): il modello pronto per l'inferenza e il dizionario del
        checkpoint con `config`, `num_features`, `best_val_loss`, `best_epoch`, `seed`.
    """
    ckpt = torch.load(Path(path), map_location=device, weights_only=True)
    version = ckpt.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Formato checkpoint {version!r} non supportato (atteso {CHECKPOINT_FORMAT_VERSION}).")

    model = build_model(ckpt["config"], ckpt["num_features"])
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), ckpt


# =============================================================================
# 5. Dati sintetici per provare il training senza i file di DATA/
# =============================================================================


def make_dummy_loaders(
    cfg: Mapping[str, Any],
    *,
    n_train_batches: int = 8,
    n_val_batches: int = 2,
    batch_size: int = 64,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Liste di batch sintetici conformi al contratto, usabili come train e val loader.

    I target sono casuali e non legati agli input: servono a verificare che il
    ciclo di addestramento funzioni (loss finite, salvataggi, early stopping),
    non a ottenere un modello utile.
    """
    num_features = infer_num_features(cfg)
    train = [make_dummy_batch(batch_size, num_features=num_features, seed=seed + i) for i in range(n_train_batches)]
    val = [make_dummy_batch(batch_size, num_features=num_features, seed=seed + 1000 + i) for i in range(n_val_batches)]
    return train, val


# =============================================================================
# 6. Addestramento completo
# =============================================================================


@dataclass
class TrainResult:
    """Esito di `train()`: il modello con i pesi migliori e il resoconto dell'addestramento."""

    model: nn.Module
    history: list[dict[str, Any]]
    best_epoch: int
    best_val_loss: float
    stopped_early: bool
    output_dir: Path | None = None
    checkpoint_path: Path | None = field(default=None)


def default_output_dir(cfg: Mapping[str, Any]) -> Path:
    """Cartella degli output: `<output.dir>/<output.experiment_name>` dentro il progetto."""
    out = cfg.get("output", {})
    return PROJECT_ROOT / out.get("dir", "runs") / str(out.get("experiment_name", "default"))


def _write_history(path: Path, history: list[dict[str, Any]], best_epoch: int, best_val_loss: float, stopped_early: bool) -> None:
    payload = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "stopped_early": stopped_early,
        "epochs": history,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train(
    cfg: Mapping[str, Any],
    train_loader: Iterable[Mapping[str, Any]],
    val_loader: Iterable[Mapping[str, Any]],
    *,
    output_dir: str | Path | None = None,
    save: bool = True,
    verbose: bool = True,
) -> TrainResult:
    """Addestra il modello indicato in `cfg` e restituisce quello con la validazione migliore.

    Args:
        cfg: configurazione (config.yaml). Legge `seed`, `model`, `training`, `output`.
        train_loader, val_loader: iterabili di batch conformi al contratto
            (DataLoader veri o liste di batch sintetici). Devono poter essere
            percorsi più volte, una per epoca.
        output_dir: cartella dei file di output; se None usa `default_output_dir(cfg)`.
        save: se False non scrive nulla su disco (utile nei test).
        verbose: stampa una riga per epoca.

    I modelli senza parametri (la persistenza) non hanno nulla da ottimizzare:
    viene misurata la loss una sola volta e salvato comunque il checkpoint, così
    la valutazione può trattarli come qualsiasi altro modello.
    """
    cfg = copy.deepcopy(dict(cfg))
    tcfg = cfg["training"]

    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(tcfg.get("device", "auto"))

    num_features = infer_num_features(cfg)
    model = build_model(cfg, num_features).to(device)
    criterion = build_loss(cfg).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    trainable = len(params) > 0
    epochs = int(tcfg["epochs"]) if trainable else 1
    if epochs < 1:
        raise ValueError("training.epochs deve essere almeno 1.")

    optimizer = build_optimizer(params, tcfg) if trainable else None
    scheduler = build_scheduler(optimizer, tcfg, epochs) if trainable else None
    grad_clip = tcfg.get("gradient_clip")
    grad_clip = float(grad_clip) if grad_clip is not None else None
    patience = tcfg.get("early_stopping_patience")
    patience = int(patience) if patience is not None and int(patience) > 0 else None

    out_dir = Path(output_dir) if output_dir is not None else default_output_dir(cfg)
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / CONFIG_USED_NAME).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    checkpoint_path = out_dir / CHECKPOINT_NAME

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Modello: {cfg['model']['name']} ({n_params:,} parametri) | device: {device} | epoche: {epochs}")

    history: list[dict[str, Any]] = []
    best_val = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, epochs + 1):
        start = time.time()
        lr = optimizer.param_groups[0]["lr"] if optimizer is not None else 0.0

        # Training (o sola misura, per i modelli senza parametri) e validazione.
        train_loss = run_epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer, grad_clip=grad_clip, check_contract=(epoch == 1),
        )
        val_loss = run_epoch(model, val_loader, criterion, device, check_contract=(epoch == 1))

        # Salviamo il modello solo se la validazione migliora (early stopping + miglior epoca).
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            if save:
                save_checkpoint(checkpoint_path, best_state, cfg, num_features, best_val, best_epoch)
        else:
            epochs_without_improvement += 1

        # Lo scheduler plateau guarda la validazione; gli altri avanzano di un'epoca.
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr, "seconds": round(time.time() - start, 2)}
        )
        stopped_early = patience is not None and epochs_without_improvement >= patience
        if save:
            _write_history(out_dir / HISTORY_NAME, history, best_epoch, best_val, stopped_early)

        if verbose:
            marker = " *" if best_epoch == epoch else ""
            print(
                f"epoca {epoch:3d}/{epochs} | train {train_loss:.5f} | val {val_loss:.5f} | "
                f"lr {lr:.2e} | {history[-1]['seconds']:.1f}s{marker}"
            )
        if stopped_early:
            if verbose:
                print(f"Early stopping: nessun miglioramento per {patience} epoche (migliore: epoca {best_epoch}).")
            break

    # Il modello restituito ha i pesi dell'epoca migliore, non dell'ultima.
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    return TrainResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_val_loss=best_val,
        stopped_early=stopped_early,
        output_dir=out_dir if save else None,
        checkpoint_path=checkpoint_path if save else None,
    )
