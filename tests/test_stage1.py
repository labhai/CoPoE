"""Stage-1 checkpoint and frozen-embedding smoke checks."""

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from copoe.encode import run as export_embeddings
from copoe.encoders import build_encoder
from copoe.pretrain import run as pretrain
from copoe.stage1 import load_stage1


class Stage1Tests(unittest.TestCase):
    def test_pretraining_rejects_overlap_between_byte_and_unicode_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(root / "train.npz", ids=np.array([b"subject_a"]),
                     x=np.zeros((1, 4)), y=np.array([0]))
            np.savez(root / "val.npz", ids=np.array(["subject_a"]),
                     x=np.zeros((1, 4)), y=np.array([0]))
            args = argparse.Namespace(modality="clinical", train=root / "train.npz",
                                      val=root / "val.npz", out=root / "run",
                                      epochs=1, batch_size=1, lr=1e-3)
            with self.assertRaisesRegex(ValueError, "overlapping subjects"):
                pretrain(args)

    def test_rejects_fractional_diagnosis_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.npz"
            np.savez(path, ids=np.array(["a"]), x=np.zeros((1, 4)), y=np.array([0.5]))
            with self.assertRaisesRegex(ValueError, "y must contain"):
                load_stage1(path, "clinical")

    def test_architecture_shapes(self):
        for modality, shape in (("image", (1, 16, 16, 16)),
                                ("genomic", (12,)), ("clinical", (8,)),
                                ("biospecimen", (5,))):
            with self.subTest(modality=modality):
                kwargs = {"hidden_dims": (16, 8)} if modality == "genomic" else {}
                if modality == "image":
                    kwargs["base_channels"] = 8
                model = build_encoder(modality, shape, **kwargs).eval()
                with torch.no_grad():
                    self.assertEqual(tuple(model.forward_features(torch.zeros((2,) + shape)).shape), (2, 128))

    def test_pretrain_and_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for modality in ("clinical", "genomic"):
                with self.subTest(modality=modality):
                    for split, count in (("train", 9), ("val", 6)):
                        rng = np.random.default_rng(42 + count)
                        x = rng.normal(size=(count, 8)).astype(np.float32)
                        y = np.arange(count) % 3
                        np.savez_compressed(root / f"{modality}_{split}.npz",
                                            ids=np.array([f"{split}_{i}" for i in range(count)]), x=x, y=y)
                    out = root / modality
                    args = argparse.Namespace(
                        modality=modality, train=root / f"{modality}_train.npz",
                        val=root / f"{modality}_val.npz", out=out, device="cpu", seed=42,
                        epochs=1, patience=1, batch_size=3, lr=1e-3, weight_decay=0.0,
                        label_smoothing=0.0, monitor="val_loss", balanced_sampler=False,
                        class_weighted_loss=False, scheduler="none", warmup_epochs=0,
                        grad_clip=1.0, embed_dim=128, dropout=0.0, base_channels=8,
                        hidden_dim=16, depth=1, genomic_hidden_dims=(16, 8),
                        recon_weight=.2, kl_beta=.02)
                    pretrain(args)
                    export_embeddings(argparse.Namespace(
                        modality=modality, checkpoint=out / "best_encoder.pt",
                        input=root / f"{modality}_val.npz", output=root / f"{modality}_out.npz",
                        batch_size=3, device="cpu", trusted_legacy=False))
                    with np.load(root / f"{modality}_out.npz", allow_pickle=False) as data:
                        self.assertEqual(data["embedding"].shape, (6, 128))
                        self.assertTrue(np.isfinite(data["embedding"]).all())

    def test_rejects_object_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.npz"
            np.savez_compressed(path, ids=np.array(["a"], dtype=object),
                                x=np.zeros((1, 4)), y=np.array([0]))
            with self.assertRaises(ValueError):
                load_stage1(path, "clinical")

    def test_directory_memmap_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / "ids.npy", np.array(["a", "b"]))
            np.save(root / "x.npy", np.zeros((2, 1, 16, 16, 16), dtype=np.float32))
            np.save(root / "y.npy", np.array([0, 1]))
            ids, x, y, missing = load_stage1(root, "image")
            self.assertEqual(ids.tolist(), ["a", "b"])
            self.assertEqual(x.shape, (2, 1, 16, 16, 16))
            self.assertEqual(y.tolist(), [0, 1])
            self.assertIsNone(missing)


if __name__ == "__main__":
    unittest.main()
