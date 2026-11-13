import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from power_risk.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig, OutputConfig
from power_risk.data import _split_events, load_risk_dataset
from run_ablation_study import (ABLATION_VARIANTS, NullRiskLibrary, parse_variant_names,
                                train_ablation_one_epoch)


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3, 5)

    def forward(self, x, sources):
        return self.layer(x), x * self.layer.weight.sum(), None


class SpyLibrary:
    def __init__(self):
        self.calls = 0

    def contrastive_loss(self, embeddings, labels, temperature):
        self.calls += 1
        return embeddings.square().mean()


class AblationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10)
        self.config = ExperimentConfig(DataConfig(), ModelConfig(),
                                       TrainConfig(pseudo_label_threshold=0.0), OutputConfig())
        self.x = torch.randn(6, 3)
        self.labels = torch.tensor([0, 1, 2, 3, 4, 1])
        self.mask = torch.tensor([True, True, True, False, False, False])

    def train(self, model, labels=None, semi=True, contrastive=True, library=True):
        data = TensorDataset(self.x, torch.zeros(6), torch.zeros(6),
                             self.labels if labels is None else labels, self.mask)
        spy, stats = SpyLibrary(), {}
        train_ablation_one_epoch(model, DataLoader(data, batch_size=6),
                                 torch.optim.SGD(model.parameters(), lr=.01),
                                 nn.CrossEntropyLoss(), spy, self.config, torch.device("cpu"),
                                 library, semi, contrastive, stats)
        return spy, stats

    def test_no_contrastive_keeps_pseudo_loss(self):
        spy, stats = self.train(TinyNet(), contrastive=False)
        self.assertEqual(spy.calls, 0)
        self.assertEqual(stats["pseudo_samples"], 3)

    def test_no_semi_keeps_contrastive(self):
        spy, stats = self.train(TinyNet(), semi=False)
        self.assertEqual(spy.calls, 1)
        self.assertEqual(stats["pseudo_samples"], 0)
        self.assertEqual(stats["labeled_samples"], 3)

    def test_unlabeled_ground_truth_not_used(self):
        a = TinyNet()
        b = copy.deepcopy(a)
        changed = self.labels.clone()
        changed[~self.mask] = 0
        self.train(a, semi=False)
        self.train(b, labels=changed, semi=False)
        for left, right in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_no_library_disables_proto_loss(self):
        spy, stats = self.train(TinyNet(), library=False)
        self.assertEqual(spy.calls, 0)
        self.assertEqual(stats["pseudo_samples"], 3)

    def test_confidence_fallback_and_legacy_behavior(self):
        x, confidence = torch.zeros(2, 3), torch.tensor([.3, .8])
        self.assertEqual(NullRiskLibrary(True).unknown_flags(x, confidence, .55).tolist(), [True, False])
        self.assertFalse(NullRiskLibrary().unknown_flags(x, confidence, .55).any())
        names = parse_variant_names("Full,no_prototype,no_contrastive,no_semi")
        self.assertEqual(len(names), 4)
        self.assertTrue(ABLATION_VARIANTS["No-Contrastive"]["use_library"])

    def test_group_split_excludes_unknown_and_overlap(self):
        frame = pd.DataFrame({"time": np.repeat(np.arange(100), 4)})
        frame["__y_class"] = frame.time % 5
        frame["__y_unknown"] = frame.time >= 90
        cfg = DataConfig(split_group_column="time")
        train, val, test = _split_events(frame, cfg)
        sets = [set(f.time) for f in [train, val, test]]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(set.union(*sets), set(range(100)))
        self.assertFalse(train["__y_unknown"].any())
        self.assertFalse(val["__y_unknown"].any())
        self.assertEqual(int(test["__y_unknown"].sum()), 40)
        cfg.split_group_column = "missing"
        with self.assertRaises(ValueError):
            _split_events(frame, cfg)

    def test_group_id_and_label_metadata_not_encoded(self):
        with tempfile.TemporaryDirectory() as temp:
            frame = pd.DataFrame({"time": np.arange(100), "marker": np.arange(100) % 2,
                                  "value": np.linspace(0, 1, 100),
                                  "label": np.arange(100) % 2,
                                  "type": np.where(np.arange(100) % 2, "dos", "normal")})
            frame.to_csv(Path(temp) / "network.csv", index=False)
            cfg = DataConfig(data_dir=temp, auto_download=False, allow_synthetic_fallback=False,
                             split_group_column="time", excluded_feature_columns=["marker"],
                             sequence_length=2, stride=2)
            bundle = load_risk_dataset(cfg)
            self.assertNotIn("marker", bundle.split_summary["encoded_columns"])
            self.assertNotIn("time", bundle.split_summary["encoded_columns"])
            self.assertTrue(all(v == 0 for v in bundle.split_summary["group_overlaps"].values()))


if __name__ == "__main__":
    unittest.main()
