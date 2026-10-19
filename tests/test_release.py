"""Release checks for masking, checkpoint IO, and the fusion CLI."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from copoe.config import attach_paper_heads, paper_config
from copoe.data import load_split
from copoe.evaluate import load_model, subset_masks
from copoe.model import CCBPoEHead
from copoe.train import run_seed, validation_score


class ReleaseTests(unittest.TestCase):
    def test_poe_matches_analytic_masked_gaussian_product(self):
        model = CCBPoEHead(paper_config(42)).eval()
        means = torch.arange(1.0, 5.0).view(1, 4, 1).expand(15, 4, 48)
        logvars = torch.zeros_like(means)
        masks = torch.tensor(subset_masks(), dtype=torch.float32)
        with torch.no_grad():
            fused = model._fuse_poe(means, logvars, masks)
        expected_precision = 1.0 + masks.sum(dim=1, keepdim=True)
        expected_mean = (means * masks.unsqueeze(-1)).sum(dim=1) / expected_precision
        expected_variance = expected_precision.reciprocal().expand_as(expected_mean)
        torch.testing.assert_close(fused["mu_fuse"], expected_mean, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(fused["logvar_fuse"].exp(), expected_variance, rtol=1e-5, atol=1e-6)

    def test_input_validation_rejects_misaligned_or_invalid_targets(self):
        valid = {"x": np.zeros((3, 512)), "mask": np.ones((3, 4)), "y": np.arange(3)}
        invalid = (
            {"x": np.zeros((4, 512))},
            {"y": np.arange(3).reshape(3, 1)},
            {"y": np.array([0.0, 1.5, 2.0])},
            {"amy_label": np.array([0, 1, 0])},
            {"amy_label": np.array([0, 1, 0]), "amy_mask": np.array([1, 2, 1])},
            {"amy_label": np.array([0, 1, 0]), "amy_mask": np.array([1, np.nan, 1])},
            {"amy_label": np.array([0, 0.5, 1]), "amy_mask": np.ones(3)},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.npz"
            for overrides in invalid:
                with self.subTest(overrides=list(overrides)):
                    np.savez(path, **(valid | overrides))
                    with self.assertRaises(ValueError):
                        load_split(path)
            np.savez(path, **valid, amy_label=np.array([0, np.nan, 1]), amy_mask=np.array([1, 0, 1]))
            self.assertEqual(load_split(path)["amy_label"].tolist(), [0, -1, 1])

    def test_original_checkpoint_selection_criterion_is_preserved(self):
        class FixedPredictions(torch.nn.Module):
            enable_biology_heads = True

            def forward(self, x, mask, return_dict=False):
                self.last_mask = mask
                return {"logits": x[:, :3], "amy_logit": x[:, 3], "tau_logit": x[:, 4]}

        model = FixedPredictions()
        data = {
            "x": np.array([[4, 0, 0, -1, 1], [0, 4, 0, 1, -1], [0, 0, 4, -1, 1]], dtype=np.float32),
            "mask": np.ones((3, 4), dtype=np.float32), "y": np.arange(3),
            "amy_label": np.array([0, 1, 0]), "tau_label": np.array([1, 0, 1]),
            "amy_mask": np.ones(3), "tau_mask": np.ones(3),
        }
        score, _ = validation_score(model, data, torch.device("cpu"), paper_selection=True)
        self.assertEqual(score, 2.0)  # BAcc + 0.5 amy AUROC + 0.5 tau AUROC.
        score, _ = validation_score(model, data, torch.device("cpu"), paper_selection=False)
        self.assertEqual(score, 1.0)
        torch.testing.assert_close(model.last_mask, torch.ones(3, 4))

    def test_run_rejects_overwrite_and_nonpositive_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with self.assertRaisesRegex(ValueError, "positive"):
                run_seed(42, {}, {}, out, torch.device("cpu"), 0, 1, 2, True)
            (out / "best.pt").write_bytes(b"existing checkpoint")
            with self.assertRaises(FileExistsError):
                run_seed(42, {}, {}, out, torch.device("cpu"), 1, 1, 2, True)

    def test_missing_expert_does_not_affect_prediction(self):
        torch.manual_seed(42)
        model = CCBPoEHead(paper_config(42)).eval()
        attach_paper_heads(model)
        x = torch.randn(4, 512)
        mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]]).repeat(4, 1)
        changed = x.clone()
        changed[:, 384:] = 1000.0
        with torch.no_grad():
            one = model(x, mask, return_dict=True)
            two = model(changed, mask, return_dict=True)
        torch.testing.assert_close(one["concept_mean"], two["concept_mean"], rtol=0, atol=0)
        torch.testing.assert_close(one["logits"], two["logits"], rtol=0, atol=0)
        torch.testing.assert_close(one["amy_logit"], two["amy_logit"], rtol=0, atol=0)

    def test_training_and_checkpoint_round_trip(self):
        rng = np.random.default_rng(42)

        def make_split(n, path):
            x = rng.normal(size=(n, 512)).astype(np.float32)
            mask = np.ones((n, 4), dtype=np.float32)
            y = np.arange(n, dtype=np.int64) % 3
            amy = np.arange(n, dtype=np.int64) % 2
            tau = (np.arange(n, dtype=np.int64) // 2) % 2
            np.savez(path, x=x, mask=mask, y=y, amy_label=amy, amy_mask=np.ones(n, dtype=np.float32),
                     tau_label=tau, tau_mask=np.ones(n, dtype=np.float32))

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            make_split(18, tmp / "train.npz")
            make_split(12, tmp / "val.npz")
            train, val = load_split(tmp / "train.npz"), load_split(tmp / "val.npz")
            result = run_seed(42, train, val, tmp / "run", torch.device("cpu"),
                              epochs=10, patience=3, batch_size=18, paper_selection=True)
            self.assertEqual(result["best_epoch"], 10)
            model = load_model(tmp / "run/best.pt", torch.device("cpu"))
            self.assertEqual(model.cfg.group_dims, (8, 16, 16, 8))


if __name__ == "__main__":
    unittest.main()
