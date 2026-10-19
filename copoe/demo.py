"""Run the complete public CLI on small synthetic, non-ADNI embeddings."""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

from .config import MODALITY_ORDER
from .prepare import prepare_splits


def make_synthetic_inputs(out_dir: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(7)
    embeddings_dir = out_dir / "embeddings"
    embeddings_dir.mkdir(parents=True)
    pet_csv = out_dir / "synthetic_pet.csv"
    with pet_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("id", "amy", "tau"))
        for split, count in (("train", 120), ("val", 45), ("test", 45)):
            ids = np.array([f"{split}_{i:04d}" for i in range(count)])
            labels = np.tile(np.arange(3), count // 3)
            rng.shuffle(labels)
            observed = rng.random((count, 4)) > 0.2 if split == "train" else np.ones((count, 4), dtype=bool)
            observed[observed.sum(axis=1) == 0, 0] = True
            for i, identifier in enumerate(ids):
                amy = int(labels[i] > 0) ^ int(rng.random() < 0.1)
                tau = int(labels[i] == 2) ^ int(rng.random() < 0.1)
                writer.writerow((identifier, amy, tau))
            for col, modality in enumerate(MODALITY_ORDER):
                embedding = rng.normal(size=(count, 128)).astype(np.float32)
                embedding[:, :8] += (labels[:, None] - 1) * (0.4 + 0.1 * col)
                np.savez_compressed(embeddings_dir / f"{modality}_{split}.npz",
                                    ids=ids[observed[:, col]], y=labels[observed[:, col]],
                                    embedding=embedding[observed[:, col]])
    return embeddings_dir, pet_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/demo"))
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"{args.out} is not empty; choose a fresh output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    embeddings_dir, pet_csv = make_synthetic_inputs(args.out)
    data_dir = args.out / "data"
    prepare_splits(embeddings_dir, data_dir, pet_csv)
    run_dir = args.out / "model"
    subprocess.run([sys.executable, "-m", "copoe.train", "--train", str(data_dir / "train.npz"),
                    "--val", str(data_dir / "val_all4.npz"), "--test", str(data_dir / "test_all4.npz"),
                    "--out", str(run_dir), "--seeds", "42", "--epochs", str(args.epochs),
                    "--device", "cpu"], check=True)
    subprocess.run([sys.executable, "-m", "copoe.evaluate", "--checkpoint", str(run_dir / "seed42/best.pt"),
                    "--data", str(data_dir / "test_all4.npz"), "--output", str(run_dir / "standalone_eval.csv"),
                    "--device", "cpu"], check=True)
    print(f"Synthetic smoke run completed: {run_dir / 'standalone_eval.csv'}")


if __name__ == "__main__":
    main()
