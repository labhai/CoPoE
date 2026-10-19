"""Assemble frozen modality embeddings into CoPoE fusion splits."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .config import MODALITY_ORDER


def _subject_id(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_modality(path: Path, trusted_object_ids: bool) -> dict[str, tuple[int, np.ndarray]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=trusted_object_ids) as archive:
        if not {"ids", "y", "embedding"}.issubset(archive.files):
            raise ValueError(f"{path}: expected ids, y, embedding arrays")
        try:
            ids = archive["ids"]
        except ValueError as exc:
            raise ValueError(f"{path}: object-typed IDs require --trusted-object-ids") from exc
        labels = np.asarray(archive["y"])
        embeddings = np.asarray(archive["embedding"], dtype=np.float32)
    if ids.ndim != 1 or labels.shape != ids.shape or embeddings.shape != (len(ids), 128):
        raise ValueError(f"{path}: expected ids/y [N] and embedding [N,128]")
    if not np.isfinite(embeddings).all() or not np.isin(labels, (0, 1, 2)).all():
        raise ValueError(f"{path}: embeddings must be finite and labels must be CN=0/MCI=1/AD=2")
    result = {}
    for identifier, label, embedding in zip(ids, labels, embeddings):
        key = _subject_id(identifier)
        if not key or key in result:
            raise ValueError(f"{path}: empty or duplicate subject ID")
        result[key] = (int(label), embedding)
    return result


def _read_pet(path: Path | None) -> dict[str, tuple[int, int]]:
    if path is None:
        return {}
    result = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not {"id", "amy", "tau"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: expected id,amy,tau columns")
        for row in reader:
            key = row["id"].strip()
            if not key or key in result:
                raise ValueError(f"{path}: empty or duplicate subject ID")
            values = tuple(-1 if not row[name].strip() else int(row[name]) for name in ("amy", "tau"))
            if not all(value in (-1, 0, 1) for value in values):
                raise ValueError(f"{path}: PET labels must be blank, 0, or 1")
            result[key] = values
    return result


def prepare_splits(
    embeddings_dir: Path,
    out_dir: Path,
    pet_labels: Path | None = None,
    trusted_object_ids: bool = False,
) -> dict:
    splits = {
        split: {
            modality: _read_modality(embeddings_dir / f"{modality}_{split}.npz", trusted_object_ids)
            for modality in MODALITY_ORDER
        }
        for split in ("train", "val", "test")
    }
    all_ids = {split: set().union(*(data.keys() for data in modalities.values())) for split, modalities in splits.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if all_ids[left] & all_ids[right]:
            raise ValueError(f"Participant leakage between {left} and {right} splits")

    means, stds = [], []
    for modality in MODALITY_ORDER:
        if not splits["train"][modality]:
            raise ValueError(f"train: no observed {modality} embeddings")
        values = np.stack([embedding for _, embedding in splits["train"][modality].values()])
        mean, std = values.mean(axis=0), values.std(axis=0)
        means.append(mean)
        stds.append(np.where(std > 0, std, 1.0))
    pet = _read_pet(pet_labels)
    manifest = {"modality_order": list(MODALITY_ORDER), "embedding_dim": 128, "splits": {}}
    generated = {}
    for split, modalities in splits.items():
        ids = sorted(set.union(*(set(data) for data in modalities.values()))) if split == "train" else sorted(set.intersection(*(set(data) for data in modalities.values())))
        if not ids:
            raise ValueError(f"{split}: no subjects with the required modalities")
        x = np.zeros((len(ids), 512), dtype=np.float32)
        mask = np.zeros((len(ids), 4), dtype=np.float32)
        y = np.empty(len(ids), dtype=np.int64)
        amy_label = np.full(len(ids), -1, dtype=np.int64)
        tau_label = np.full(len(ids), -1, dtype=np.int64)
        amy_mask = np.zeros(len(ids), dtype=np.float32)
        tau_mask = np.zeros(len(ids), dtype=np.float32)
        for row, identifier in enumerate(ids):
            labels = set()
            for col, modality in enumerate(MODALITY_ORDER):
                item = modalities[modality].get(identifier)
                if item is None:
                    continue
                label, embedding = item
                labels.add(label)
                x[row, col * 128:(col + 1) * 128] = (embedding - means[col]) / stds[col]
                mask[row, col] = 1.0
            if len(labels) != 1:
                raise ValueError(f"{split}: conflicting diagnosis labels for one subject")
            y[row] = labels.pop()
            amy_label[row], tau_label[row] = pet.get(identifier, (-1, -1))
            amy_mask[row] = float(amy_label[row] >= 0)
            tau_mask[row] = float(tau_label[row] >= 0)
        filename = "train.npz" if split == "train" else f"{split}_all4.npz"
        generated[filename] = dict(x=x, mask=mask, y=y, amy_label=amy_label,
                                   amy_mask=amy_mask, tau_label=tau_label, tau_mask=tau_mask)
        manifest["splits"][split] = {
            "subjects": len(ids),
            "diagnosis_counts": [int((y == cls).sum()) for cls in range(3)],
            "amy_labeled": int(amy_mask.sum()),
            "tau_labeled": int(tau_mask.sum()),
            "observed_modalities": [int(mask[:, col].sum()) for col in range(4)],
        }
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"{out_dir} is not empty; choose a fresh output directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "normalizer.npz", mean=np.stack(means), std=np.stack(stds))
    for filename, arrays in generated.items():
        np.savez_compressed(out_dir / filename, **arrays)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--pet-labels", type=Path, help="Optional CSV with id,amy,tau columns")
    parser.add_argument("--trusted-object-ids", action="store_true", help="Allow pickle-based IDs in trusted local NPZ files")
    args = parser.parse_args()
    print(json.dumps(prepare_splits(args.embeddings_dir, args.out, args.pet_labels, args.trusted_object_ids), indent=2))


if __name__ == "__main__":
    main()
