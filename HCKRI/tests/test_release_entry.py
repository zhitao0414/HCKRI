import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from run_hckri import clean, metrics_from_saved, resolve, save_json, switches


class ReleaseEntryTests(unittest.TestCase):
    def test_json_has_no_nonstandard_nan(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "result.json"
            save_json(path, {"value": np.float64(.5), "missing": float("nan"), "array": np.array([1, 2])})
            value = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(value, {"value": .5, "missing": None, "array": [1, 2]})

    def test_learning_controls_are_independent(self):
        self.assertFalse(switches("H")["use_library"])
        self.assertFalse(switches("H")["use_contrastive"])
        self.assertFalse(switches("I")["use_contrastive"])
        self.assertTrue(switches("I")["use_library"])
        self.assertFalse(switches("J")["use_semi"])
        self.assertTrue(switches("J")["use_library"])

    def test_relative_paths_resolve_against_repository(self):
        self.assertEqual(resolve(Path("configs/smoke.json")), ROOT / "configs/smoke.json")

    def test_uarc_is_not_explicit_unknown_recall(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "predictions.npz"
            np.savez(path, y_true=[0, 1, 2, 3, 4], pred=[0, 1, 2, 4, 4],
                     unknown_attack_mask=[0, 0, 1, 0, 1], unknown_flags=[1, 0, 0, 0, 1])
            metrics, matrix = metrics_from_saved(path)
            self.assertEqual(metrics["uarc"], 1.0)
            self.assertEqual(metrics["unknown_flag_recall"], .5)
            self.assertEqual(metrics["unknown_flag_precision"], .5)
            self.assertEqual(matrix.shape, (5, 5))
            self.assertEqual(matrix.sum(), 5)

    def test_empty_unknown_population_remains_undefined(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "predictions.npz"
            np.savez(path, y_true=[0, 1], pred=[0, 1], unknown_attack_mask=[0, 0], unknown_flags=[0, 0])
            metrics, _ = metrics_from_saved(path)
            self.assertIsNone(clean(metrics)["uarc"])


if __name__ == "__main__":
    unittest.main()
