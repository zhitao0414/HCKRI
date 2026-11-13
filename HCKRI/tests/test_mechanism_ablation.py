import copy
import sys
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from power_risk.ablations import AblationRiskNet
from power_risk.config import ModelConfig
from power_risk.mechanism_ablations import MechanismRiskNet
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_mechanism_ablation import CASES, make_config
from power_risk import load_experiment_config


class MechanismTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        self.cfg = ModelConfig(model_dim=16, lstm_hidden=8, source_embedding_dim=4,
                               transformer_heads=2, transformer_layers=1, dropout=0)
        self.x = torch.randn(3, 12, 9)
        self.sources = torch.arange(12).remainder(3).repeat(3, 1)

    def model(self, mechanism="Full", config=None):
        return MechanismRiskNet(9, 3, 5, config or self.cfg, mechanism=mechanism).eval()

    def test_full_exactly_matches_historical_full(self):
        legacy = AblationRiskNet(9, 3, 5, self.cfg).eval()
        candidate = self.model()
        candidate.load_state_dict(legacy.state_dict())
        with torch.no_grad():
            for a, b in zip(legacy(self.x, self.sources), candidate(self.x, self.sources)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_single_scale_parameter_count_matches(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.cnn_kernel_sizes = [5, 5, 5]
        a, b = self.model(), self.model(config=cfg)
        self.assertEqual(sum(p.numel() for p in a.parameters()),
                         sum(p.numel() for p in b.parameters()))
        self.assertEqual([c.kernel_size for c in b.convs], [(5,), (5,), (5,)])

    def test_full_training_dropout_matches(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.dropout = 0.2
        legacy = AblationRiskNet(9, 3, 5, cfg).train()
        candidate = self.model(config=cfg).train()
        candidate.load_state_dict(legacy.state_dict())
        torch.manual_seed(99)
        expected = legacy(self.x, self.sources)
        torch.manual_seed(99)
        actual = candidate(self.x, self.sources)
        for a, b in zip(expected, actual):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_controls_remain_sample_dependent(self):
        base = self.model()
        for name in ["No-CNN-Residual", "Fixed-Source-Gate", "No-Source-Context",
                     "Final-Stage-Only"]:
            candidate = self.model(name)
            candidate.load_state_dict(base.state_dict())
            out, embedding, _ = candidate(self.x, self.sources)
            self.assertEqual(out.shape, (3, 5))
            self.assertTrue(torch.isfinite(out).all())
            self.assertGreater(float((embedding[0]-embedding[1]).abs().max()), 1e-6)
            self.assertFalse(torch.equal(out, base(self.x, self.sources)[0]))

    def test_removed_paths_do_not_receive_gradient(self):
        cases = [("Fixed-Source-Gate", "gate."),
                 ("No-Source-Context", "source_projection.")]
        for name, prefix in cases:
            model = self.model(name)
            model(self.x, self.sources)[0].square().sum().backward()
            self.assertTrue(all(p.grad is None for n, p in model.named_parameters()
                                if n.startswith(prefix)))

    def test_final_stage_only_zeroes_only_early_fusion_inputs(self):
        model = self.model("Final-Stage-Only")
        captured = []
        handle = model.stage_fusion.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach()))
        model(self.x, self.sources)
        handle.remove()
        self.assertEqual(int(torch.count_nonzero(captured[0][:, :32])), 0)
        self.assertGreater(int(torch.count_nonzero(captured[0][:, 32:])), 0)

    def test_training_settings_match_historical_configs(self):
        root = Path(__file__).resolve().parents[1]
        for dataset, (filename, _) in CASES.items():
            original = load_experiment_config(root / "configs" / filename)
            actual = make_config(dataset, 20260900, "ag")
            self.assertEqual(asdict(actual.model), asdict(original.model))
            expected_train = asdict(original.train)
            expected_train["seed"] = 20260900
            self.assertEqual(asdict(actual.train), expected_train)
            for key in ["sequence_length", "stride", "val_size", "test_size", "labeled_fraction"]:
                self.assertEqual(getattr(actual.data, key), getattr(original.data, key))
            self.assertFalse(actual.data.allow_synthetic_fallback)
        psad = make_config("psad", 20260900, "ag")
        self.assertEqual(psad.data.split_group_column, "time")
        self.assertEqual(psad.data.excluded_feature_columns, ["marker", "time", "dataset_part"])

    def test_multiview_keeps_ag_training_and_window_settings(self):
        for dataset in ["ton_iot", "edge_iiot"]:
            ag = make_config(dataset, 20260900, "ag")
            view = make_config(dataset, 20260900, "multiview")
            self.assertEqual(asdict(ag.model), asdict(view.model))
            self.assertEqual(asdict(ag.train), asdict(view.train))
            self.assertEqual(ag.data.sequence_length, view.data.sequence_length)
            self.assertEqual(ag.data.stride, view.data.stride)


if __name__ == "__main__":
    unittest.main()
