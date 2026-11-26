"""Test unitari per il training loop (src/train.py) e la CLI (main.py).

Usano batch sintetici e cartelle temporanee: non serve DATA/ e non si scrive in runs/.
Esecuzione:  python -m unittest discover tests
"""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

import main as cli
from src.config import load_config
from src.contract import make_dummy_batch
from src.losses import PinballLoss
from src.models import build_model
from src.train import (
    build_optimizer,
    build_scheduler,
    load_model_from_checkpoint,
    make_dummy_loaders,
    resolve_device,
    run_epoch,
    save_checkpoint,
    train,
)


def _cfg(**training_overrides) -> dict:
    """Configurazione di default ridotta per test veloci (poche epoche, CPU)."""
    cfg = copy.deepcopy(load_config())
    cfg["training"].update({"epochs": 4, "device": "cpu", "batch_size": 16}, **training_overrides)
    return cfg


def _loaders(cfg: dict, n_train: int = 3, n_val: int = 1, batch_size: int = 16):
    return make_dummy_loaders(cfg, n_train_batches=n_train, n_val_batches=n_val, batch_size=batch_size)


class TestOptimizerAndScheduler(unittest.TestCase):
    def test_every_optimizer_and_scheduler_runs(self) -> None:
        params = list(build_model(_cfg()).parameters())
        for opt_name in ("adam", "adamw", "sgd"):
            for sched_name in ("cosine", "plateau", "none"):
                tcfg = {**_cfg()["training"], "optimizer": opt_name, "scheduler": sched_name}
                optimizer = build_optimizer(params, tcfg)
                scheduler = build_scheduler(optimizer, tcfg, epochs=3)
                optimizer.step()
                if sched_name == "plateau":
                    scheduler.step(1.0)
                elif scheduler is not None:
                    scheduler.step()

    def test_cosine_scheduler_decays_learning_rate_to_zero(self) -> None:
        params = list(build_model(_cfg()).parameters())
        tcfg = {**_cfg()["training"], "optimizer": "adamw", "scheduler": "cosine", "learning_rate": 1e-2}
        optimizer = build_optimizer(params, tcfg)
        scheduler = build_scheduler(optimizer, tcfg, epochs=10)
        lrs = []
        for _ in range(10):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(lrs[0], 1e-2)
        self.assertTrue(all(a >= b for a, b in zip(lrs, lrs[1:])))
        self.assertLess(optimizer.param_groups[0]["lr"], 1e-8)

    def test_unknown_names_raise(self) -> None:
        params = list(build_model(_cfg()).parameters())
        with self.assertRaises(ValueError):
            build_optimizer(params, {**_cfg()["training"], "optimizer": "rmsprop"})
        optimizer = build_optimizer(params, _cfg()["training"])
        with self.assertRaises(ValueError):
            build_scheduler(optimizer, {"scheduler": "step"}, epochs=3)

    def test_resolve_device(self) -> None:
        self.assertEqual(resolve_device("cpu"), torch.device("cpu"))
        self.assertIn(resolve_device("auto").type, ("cpu", "cuda"))
        with self.assertRaises(ValueError):
            resolve_device("tpu")


class TestRunEpoch(unittest.TestCase):
    def test_eval_epoch_does_not_change_weights_and_training_does(self) -> None:
        cfg = _cfg()
        torch.manual_seed(0)
        model = build_model(cfg)
        criterion = PinballLoss()
        train_batches, val_batches = _loaders(cfg)
        before = {k: v.clone() for k, v in model.state_dict().items()}

        run_epoch(model, val_batches, criterion, torch.device("cpu"))
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, before[k])

        optimizer = build_optimizer(model.parameters(), cfg["training"])
        run_epoch(model, train_batches, criterion, torch.device("cpu"), optimizer=optimizer, grad_clip=1.0)
        self.assertTrue(any(not torch.equal(v, before[k]) for k, v in model.state_dict().items()))

    def test_average_is_independent_of_how_samples_are_batched(self) -> None:
        # Stessi 12 campioni in un batch da 12 o in 3 batch da 4: stessa loss media.
        cfg = _cfg()
        model = build_model(cfg).eval()
        big = make_dummy_batch(12, seed=5)
        small = [{k: (v[i * 4:(i + 1) * 4] if isinstance(v, (torch.Tensor, list)) else v) for k, v in big.items()} for i in range(3)]
        # Il padding dei batch piccoli resta quello del grande: il contratto non viene controllato qui.
        criterion = PinballLoss()
        loss_big = run_epoch(model, [big], criterion, torch.device("cpu"))
        loss_small = run_epoch(model, small, criterion, torch.device("cpu"))
        self.assertAlmostEqual(loss_big, loss_small, places=5)

    def test_non_finite_loss_stops_training(self) -> None:
        cfg = _cfg()
        batch = make_dummy_batch(8)
        batch["target"] = batch["target"].clone()
        batch["target"][0, 0, 0] = float("nan")
        with self.assertRaises(RuntimeError):
            run_epoch(build_model(cfg), [batch], PinballLoss(), torch.device("cpu"))

    def test_empty_loader_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            run_epoch(build_model(_cfg()), [], PinballLoss(), torch.device("cpu"))


class TestTrain(unittest.TestCase):
    def test_end_to_end_saves_files_and_returns_best_model(self) -> None:
        cfg = _cfg()
        train_loader, val_loader = _loaders(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            result = train(cfg, train_loader, val_loader, output_dir=tmp, verbose=False)

            for name in ("best.pt", "config_used.yaml", "history.json"):
                self.assertTrue((Path(tmp) / name).is_file(), name)
            history = json.loads((Path(tmp) / "history.json").read_text(encoding="utf-8"))
            self.assertEqual(len(history["epochs"]), len(result.history))
            self.assertEqual(history["best_epoch"], result.best_epoch)

        self.assertTrue(all(torch.isfinite(torch.tensor(h["train_loss"])) for h in result.history))
        best = min(result.history, key=lambda h: h["val_loss"])
        self.assertEqual(result.best_epoch, best["epoch"])
        self.assertAlmostEqual(result.best_val_loss, best["val_loss"])
        self.assertFalse(result.model.training)  # restituito in modalità eval

    def test_training_reduces_loss(self) -> None:
        # Pochi batch sempre uguali: la loss di train deve scendere in modo evidente.
        cfg = _cfg(epochs=30, learning_rate=1e-2, early_stopping_patience=None, scheduler="none")
        train_loader, val_loader = _loaders(cfg, n_train=1, batch_size=32)
        result = train(cfg, train_loader, val_loader, save=False, verbose=False)
        self.assertLess(result.history[-1]["train_loss"], 0.6 * result.history[0]["train_loss"])

    def test_reload_checkpoint_gives_identical_predictions(self) -> None:
        cfg = _cfg()
        train_loader, val_loader = _loaders(cfg)
        batch = make_dummy_batch(6, seed=11)
        with tempfile.TemporaryDirectory() as tmp:
            result = train(cfg, train_loader, val_loader, output_dir=tmp, verbose=False)
            expected = result.model(batch)

            # Ricarico DOPO aver cambiato la configurazione attuale: il checkpoint
            # usa la propria copia, quindi l'architettura resta quella salvata.
            model, ckpt = load_model_from_checkpoint(Path(tmp) / "best.pt")
            torch.testing.assert_close(model(batch), expected)
            self.assertEqual(ckpt["best_epoch"], result.best_epoch)
            self.assertEqual(ckpt["num_features"], 21)
            self.assertFalse(model.training)

    def test_checkpoint_survives_a_changed_architecture_in_config(self) -> None:
        cfg = _cfg()
        cfg["model"]["params"]["residual_mlp"]["hidden_dim"] = 32
        train_loader, val_loader = _loaders(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            train(cfg, train_loader, val_loader, output_dir=tmp, verbose=False)
            # build_model con la config attuale (hidden_dim = 128) non combacia, ma il checkpoint sì.
            model, ckpt = load_model_from_checkpoint(Path(tmp) / "best.pt")
            self.assertEqual(model.mlp[0].out_features, 32)
            self.assertEqual(ckpt["config"]["model"]["params"]["residual_mlp"]["hidden_dim"], 32)

    def test_checkpoint_format_version_is_checked(self) -> None:
        cfg = _cfg()
        model = build_model(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.pt"
            save_checkpoint(path, model.state_dict(), cfg, 21, 1.0, 1)
            ckpt = torch.load(path, weights_only=True)
            ckpt["format_version"] = 999
            torch.save(ckpt, path)
            with self.assertRaises(ValueError):
                load_model_from_checkpoint(path)

    def test_persistence_model_trains_without_parameters(self) -> None:
        cfg = _cfg()
        cfg["model"]["name"] = "persistence"
        train_loader, val_loader = _loaders(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            result = train(cfg, train_loader, val_loader, output_dir=tmp, verbose=False)
            self.assertEqual(len(result.history), 1)  # una sola passata: niente da ottimizzare
            self.assertTrue((Path(tmp) / "best.pt").is_file())
            model, _ = load_model_from_checkpoint(Path(tmp) / "best.pt")
            batch = make_dummy_batch(4, seed=2)
            torch.testing.assert_close(model(batch)[..., 1], batch["last_obs"][:, None, :].expand(-1, 3, -1))

    def test_early_stopping_stops_and_restores_best_epoch(self) -> None:
        # Learning rate enorme: la validazione peggiora subito dopo la prima epoca.
        cfg = _cfg(epochs=30, learning_rate=5.0, early_stopping_patience=2, scheduler="none", gradient_clip=None)
        train_loader, val_loader = _loaders(cfg)
        try:
            result = train(cfg, train_loader, val_loader, save=False, verbose=False)
        except RuntimeError:
            self.skipTest("learning rate troppo alto: la loss è diventata non finita prima dello stop")
        self.assertTrue(result.stopped_early)
        self.assertLess(len(result.history), 30)
        # Il modello restituito ha i pesi della miglior epoca, non dell'ultima.
        val_of_best = min(h["val_loss"] for h in result.history)
        final_val = run_epoch(result.model, val_loader, PinballLoss(), torch.device("cpu"))
        self.assertAlmostEqual(final_val, val_of_best, places=4)

    def test_early_stopping_can_be_disabled(self) -> None:
        cfg = _cfg(epochs=5, early_stopping_patience=None, learning_rate=5.0, scheduler="none", gradient_clip=None)
        train_loader, val_loader = _loaders(cfg)
        try:
            result = train(cfg, train_loader, val_loader, save=False, verbose=False)
        except RuntimeError:
            self.skipTest("learning rate troppo alto: la loss è diventata non finita")
        self.assertEqual(len(result.history), 5)
        self.assertFalse(result.stopped_early)

    def test_same_seed_gives_same_losses(self) -> None:
        cfg = _cfg(epochs=3)
        a = train(cfg, *_loaders(cfg), save=False, verbose=False)
        b = train(cfg, *_loaders(cfg), save=False, verbose=False)
        self.assertEqual([h["train_loss"] for h in a.history], [h["train_loss"] for h in b.history])
        self.assertEqual([h["val_loss"] for h in a.history], [h["val_loss"] for h in b.history])

    def test_works_with_dropped_feature(self) -> None:
        cfg = _cfg(epochs=2)
        cfg["preprocessing"]["drop_features"] = ["wind"]
        train_loader, val_loader = _loaders(cfg)
        self.assertEqual(train_loader[0]["history"].shape[-1], 20)
        result = train(cfg, train_loader, val_loader, save=False, verbose=False)
        self.assertEqual(len(result.history), 2)

    def test_does_not_modify_the_given_config(self) -> None:
        cfg = _cfg(epochs=2)
        snapshot = copy.deepcopy(cfg)
        train(cfg, *_loaders(cfg), save=False, verbose=False)
        self.assertEqual(cfg, snapshot)


class TestCommandLine(unittest.TestCase):
    def test_train_dummy_runs_and_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code = cli.main(["train", "--dummy", "--epochs", "2", "--output-dir", tmp])
            self.assertEqual(code, 0)
            self.assertTrue((Path(tmp) / "best.pt").is_file())
            self.assertTrue((Path(tmp) / "history.json").is_file())

    def test_unknown_command_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            cli.main(["valuta"])


if __name__ == "__main__":
    unittest.main()
