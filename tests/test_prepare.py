"""Input-assembly checks for independent modality embeddings."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from copoe.config import MODALITY_ORDER
from copoe.data import load_split
from copoe.prepare import _read_modality, prepare_splits


class PrepareTests(unittest.TestCase):
    def test_rejects_fractional_diagnosis_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.npz"
            np.savez(path, ids=np.array(["a"]), y=np.array([0.5]), embedding=np.zeros((1, 128)))
            with self.assertRaisesRegex(ValueError, "labels must"):
                _read_modality(path, trusted_object_ids=False)

    def test_union_training_and_complete_case_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            embeddings = root / "embeddings"
            embeddings.mkdir()
            for modality in MODALITY_ORDER:
                for split, ids in (("train", ["a", "b"] if modality == "image" else ["b", "c"]),
                                   ("val", ["v"]), ("test", ["t"])):
                    labels = np.array([{"a": 0, "b": 1, "c": 2, "v": 0, "t": 2}[key] for key in ids])
                    values = np.stack([np.full(128, label + 1.0, dtype=np.float32) for label in labels])
                    np.savez_compressed(embeddings / f"{modality}_{split}.npz",
                                        ids=np.asarray(ids), y=labels, embedding=values)
            manifest = prepare_splits(embeddings, root / "out")
            train = load_split(root / "out/train.npz")
            val = load_split(root / "out/val_all4.npz")
            self.assertEqual(manifest["splits"]["train"]["subjects"], 3)
            self.assertEqual(manifest["splits"]["test"]["subjects"], 1)
            self.assertEqual(train["mask"].sum(axis=1).tolist(), [1, 4, 3])
            self.assertEqual(val["mask"].sum(axis=1).tolist(), [4])
            np.testing.assert_array_equal(train["x"][train["mask"][:, 1] == 0, 128:256], 0)

    def test_rejects_cross_split_subject(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root.mkdir(exist_ok=True)
            for modality in MODALITY_ORDER:
                for split in ("train", "val", "test"):
                    identifier = "same" if split != "test" else "other"
                    np.savez_compressed(root / f"{modality}_{split}.npz",
                                        ids=np.array([identifier]), y=np.array([0]),
                                        embedding=np.zeros((1, 128), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "leakage"):
                prepare_splits(root, root / "out")


if __name__ == "__main__":
    unittest.main()
