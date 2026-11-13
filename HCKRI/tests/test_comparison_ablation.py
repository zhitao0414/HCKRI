import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from power_risk.config import ModelConfig
from power_risk.model import MultiSourceRiskNet
from power_risk.comparison_ablations import ComparisonAblationNet


class ComparisonAblationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.cfg = ModelConfig(model_dim=16, source_embedding_dim=4, lstm_hidden=8,
                               transformer_layers=1, transformer_heads=2)
        self.x = torch.randn(3, 12, 9)
        self.s = torch.arange(12).remainder(3).repeat(3, 1)

    def test_full_initialization_and_forward_match_comparison(self):
        torch.manual_seed(9500)
        original = MultiSourceRiskNet(9, 3, 5, self.cfg)
        torch.manual_seed(9500)
        full = ComparisonAblationNet(9, 3, 5, self.cfg)
        for name, value in original.state_dict().items():
            torch.testing.assert_close(value, full.state_dict()[name], rtol=0, atol=0)
        for training in [False, True]:
            original.train(training)
            full.train(training)
            torch.manual_seed(100)
            expected = original(self.x, self.s)
            torch.manual_seed(100)
            actual = full(self.x, self.s)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_every_variant_is_finite_and_input_dependent(self):
        for letter in "ABCDEFGHIJKLMN":
            model = ComparisonAblationNet(9, 3, 5, self.cfg, variant=letter).eval()
            logits, _, _ = model(self.x, self.s)
            self.assertEqual(tuple(logits.shape), (3, 5))
            self.assertTrue(torch.isfinite(logits).all())
            self.assertGreater((logits[0] - logits[1]).abs().max().item(), 1e-6, letter)

    def test_disabled_source_paths_receive_no_gradient(self):
        for letter, prefix in [("M", "gate."), ("N", "source_projection.")]:
            model = ComparisonAblationNet(9, 3, 5, self.cfg, variant=letter)
            model(self.x, self.s)[0].square().sum().backward()
            self.assertTrue(all(p.grad is None for n, p in model.named_parameters()
                                if n.startswith(prefix)))


if __name__ == "__main__":
    unittest.main()
