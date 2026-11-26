"""Punto di ingresso del progetto.

Uso:
    python main.py train                      # addestra con config.yaml sui dati veri
    python main.py train --dummy              # prova la pipeline con batch sintetici (senza DATA/)
    python main.py train --epochs 5 --name prova1
    python main.py train --config altra.yaml --output-dir runs/mio_esperimento

La valutazione sugli split id / ood-s / ood-t (`python main.py eval`) verrà
aggiunta insieme a src/eval.py e src/metrics.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from src.config import load_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Previsione degli indici di vegetazione (NDVI, GNDVI, EVI).")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="addestra il modello indicato in config.yaml")
    train.add_argument("--config", type=Path, default=None, help="file di configurazione (default: config.yaml)")
    train.add_argument("--dummy", action="store_true", help="usa batch sintetici invece dei dati in DATA/")
    train.add_argument("--name", default=None, help="sovrascrive output.experiment_name")
    train.add_argument("--epochs", type=int, default=None, help="sovrascrive training.epochs")
    train.add_argument("--output-dir", type=Path, default=None, help="cartella esatta degli output (default: runs/<name>)")
    return parser


def _run_train(args: argparse.Namespace) -> int:
    # Import qui dentro: i comandi che non addestrano non caricano PyTorch/DataLoader.
    from src.train import make_dummy_loaders, train

    cfg = load_config(args.config) if args.config is not None else load_config()
    if args.name is not None:
        cfg["output"]["experiment_name"] = args.name
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    if args.dummy:
        train_loader, val_loader = make_dummy_loaders(cfg)
        print("Modalità --dummy: batch sintetici (i risultati non hanno significato).")
    else:
        from src.dataset import build_dataloaders

        print("Costruzione dei DataLoader (lettura di DATA/)...")
        loaders = build_dataloaders(cfg, splits=("train", "val"))
        train_loader, val_loader = loaders["train"], loaders["val"]

    result = train(cfg, train_loader, val_loader, output_dir=args.output_dir)

    print(f"\nMigliore epoca: {result.best_epoch} (val loss {result.best_val_loss:.5f})")
    print(f"Output salvati in: {result.output_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "train":
        return _run_train(args)
    raise AssertionError(f"Comando non gestito: {args.command}")  # argparse impedisce di arrivarci


if __name__ == "__main__":
    sys.exit(main())
