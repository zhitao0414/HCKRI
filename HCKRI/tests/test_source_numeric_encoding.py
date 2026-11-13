import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from power_risk.data import HashedFeatureEncoder


class SourceNumericEncodingTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "pmu": [1., 2., 3., 4.] + [np.nan] * 12,
            "protocol": ["tcp", "udp", "tcp", "udp"] + [None] * 12,
            "missing": [np.nan] * 16,
        })

    def test_legacy_behavior_unchanged(self):
        encoder = HashedFeatureEncoder(80, 32)
        encoder.fit(self.frame)
        self.assertNotIn("pmu", encoder.numeric_stats)
        self.assertIn("pmu", encoder.categorical_columns)

    def test_observed_values_preserve_numeric_type_and_statistics(self):
        encoder = HashedFeatureEncoder(80, 32, numeric_observed_only=True)
        encoder.fit(self.frame)
        self.assertIn("pmu", encoder.numeric_stats)
        self.assertIn("protocol", encoder.categorical_columns)
        self.assertNotIn("missing", encoder.numeric_stats)
        self.assertNotIn("missing", encoder.categorical_columns)
        mean, std = encoder.numeric_stats["pmu"]
        self.assertAlmostEqual(mean, 2.5)
        self.assertAlmostEqual(std, np.std([1., 2., 3., 4.]))
        x = encoder.transform(self.frame)
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(x.shape, (16, 112))
        np.testing.assert_array_equal(x[4:, :80], 0)

    def test_transform_does_not_fit_test_statistics(self):
        encoder = HashedFeatureEncoder(80, 32, numeric_observed_only=True)
        encoder.fit(self.frame)
        before = dict(encoder.numeric_stats)
        encoder.transform(pd.DataFrame({"pmu": [1e10], "protocol": ["new"]}))
        self.assertEqual(encoder.numeric_stats, before)

    def test_dirty_observations_are_not_hidden_by_missingness(self):
        frame = pd.DataFrame({"field": ["1", "bad", "bad", "2"] + [None] * 12})
        encoder = HashedFeatureEncoder(80, 32, numeric_observed_only=True)
        encoder.fit(frame)
        self.assertIn("field", encoder.categorical_columns)

    def test_real_numeric_strings_are_retained(self):
        frame = pd.DataFrame({"field": ["1", "2", "3", "4"] + [None] * 12})
        encoder = HashedFeatureEncoder(80, 32, numeric_observed_only=True)
        encoder.fit(frame)
        self.assertIn("field", encoder.numeric_stats)


if __name__ == "__main__":
    unittest.main()
