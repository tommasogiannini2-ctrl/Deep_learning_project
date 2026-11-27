"""Test delle metriche della Persona 3 su casi analitici."""

from __future__ import annotations

import unittest

import numpy as np

from src.metrics import (
    coverage,
    evaluate_predictions_by_group,
    evaluate_predictions,
    inverse_transform_targets,
    mae,
    mase,
    pinball,
    rmse,
    wmape,
)


class TestMetrics(unittest.TestCase):
    def test_point_metrics_and_mask(self) -> None:
        target = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        pred = np.array([[[2.0, 2.0], [3.0, 8.0]]])
        mask = np.array([[[False, True], [False, False]]])
        self.assertAlmostEqual(mae(pred, target, mask), 5.0 / 3.0)
        self.assertAlmostEqual(rmse(pred, target, mask), np.sqrt(17.0 / 3.0))
        self.assertAlmostEqual(wmape(pred, target, mask), 5.0 / 8.0)
        self.assertAlmostEqual(mase(pred, target, 2.0, mask), 5.0 / 6.0)

    def test_quantile_metrics_and_report(self) -> None:
        target = np.array([[[1.0], [2.0]]])
        preds = np.stack([target - 1.0, target, target + 1.0], axis=-1)
        self.assertAlmostEqual(pinball(preds, target), 0.2 / 3.0)
        self.assertAlmostEqual(coverage(preds, target), 1.0)
        report = evaluate_predictions(preds, target)
        self.assertEqual(report["point"]["mae"]["by_horizon_target"].shape, (2, 1))
        self.assertEqual(report["uncertainty"]["coverage_80"], 1.0)

    def test_invalid_quantiles_are_rejected(self) -> None:
        target = np.ones((2, 1, 1))
        preds = np.ones((2, 1, 1, 3))
        with self.assertRaises(ValueError):
            pinball(preds, target, quantiles=[0.1, 1.2, 0.9])
        with self.assertRaises(ValueError):
            evaluate_predictions(preds, target, quantiles=[0.1, 0.9, 0.95])

    def test_group_macro_report(self) -> None:
        target = np.ones((3, 1, 1))
        preds = np.stack([target - 1.0, target, target + 1.0], axis=-1)
        report = evaluate_predictions_by_group(preds, target, ["A", "A", "B"])
        self.assertEqual(report["macro"]["n_groups"], 2)
        self.assertEqual(set(report["groups"]), {"A", "B"})

    def test_inverse_scaler(self) -> None:
        scaler = {"eps": 1e-5, "target_mean": [0.5], "target_var": [0.25]}
        values = np.array([[[0.0]]])
        np.testing.assert_allclose(inverse_transform_targets(values, scaler), 0.5)


if __name__ == "__main__":
    unittest.main()
