"""Input contract for precomputed, frozen modality embeddings."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


def load_split(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"x", "mask", "y"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path}: missing arrays {sorted(missing)}")
        data = {name: archive[name].copy() for name in required}
        if data["y"].ndim != 1 or data["y"].size == 0:
            raise ValueError(f"{path}: y must be a nonempty 1-D label array")
        n = len(data["y"])
        for target in ("amy", "tau"):
            if (f"{target}_label" in archive.files) != (f"{target}_mask" in archive.files):
                raise ValueError(f"{path}: provide both {target}_label and {target}_mask")
        for name in ("amy_label", "tau_label"):
            data[name] = archive[name].copy() if name in archive.files else np.full(n, -1, dtype=np.int64)
        for name in ("amy_mask", "tau_mask"):
            data[name] = archive[name].copy() if name in archive.files else np.zeros(n, dtype=np.float32)

    x = np.asarray(data["x"], dtype=np.float32)
    mask = np.asarray(data["mask"], dtype=np.float32)
    y = np.asarray(data["y"])
    if x.shape != (len(y), 512) or mask.shape != (len(y), 4):
        raise ValueError(f"{path}: expected x [N,512], mask [N,4], y [N]")
    if not np.isfinite(x).all() or not np.isin(mask, (0, 1)).all():
        raise ValueError(f"{path}: x must be finite and mask binary")
    if not np.isin(y, (0, 1, 2)).all() or (mask.sum(axis=1) == 0).any():
        raise ValueError(f"{path}: labels must be CN=0/MCI=1/AD=2 and every row needs an observed modality")
    y = y.astype(np.int64)
    for j in range(4):
        x[mask[:, j] == 0, j * 128:(j + 1) * 128] = 0.0
    data["x"], data["mask"], data["y"] = x, mask, y
    for target in ("amy", "tau"):
        labels = np.asarray(data[f"{target}_label"])
        available = np.asarray(data[f"{target}_mask"], dtype=np.float32)
        if labels.shape != (len(y),) or available.shape != (len(y),):
            raise ValueError(f"{path}: invalid {target} label shape")
        if not np.isin(available, (0, 1)).all():
            raise ValueError(f"{path}: {target}_mask must be binary")
        if not np.isin(labels[available > 0], (0, 1)).all():
            raise ValueError(f"{path}: observed {target} labels must be 0/1")
        # Unavailable labels are never targets; canonicalize them before casting,
        # so a missing value such as NaN cannot leak into a loss calculation.
        labels = np.where(available > 0, labels, -1).astype(np.int64)
        data[f"{target}_label"], data[f"{target}_mask"] = labels, available
    return data


def loader(data: dict[str, np.ndarray], batch_size: int, seed: int, train: bool = False) -> DataLoader:
    n = len(data["y"])
    tensors = (
        torch.from_numpy(data["x"]), torch.from_numpy(data["y"]),
        torch.from_numpy(data["mask"]),
        torch.full((n, 4), float("nan")),
        torch.from_numpy(data["amy_label"]), torch.from_numpy(data["amy_mask"]),
        torch.from_numpy(data["tau_label"]), torch.from_numpy(data["tau_mask"]),
    )
    sampler = None
    if train:
        counts = np.bincount(data["y"], minlength=3).astype(np.float64)
        counts[counts == 0] = 1.0
        inverse = 1.0 / counts
        inverse /= inverse.mean()
        weights = inverse[data["y"]] * np.where(data["mask"].sum(axis=1) == 4, 2.0, 1.0)
        generator = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), n, replacement=True, generator=generator)
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=False, sampler=sampler)


def class_weights(y: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y, minlength=3).astype(np.float32)
    if (counts == 0).any():
        raise ValueError("Training split needs all three diagnosis classes")
    weights = counts.sum() / (3 * counts)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def positive_weight(data: dict[str, np.ndarray], target: str) -> float:
    y = data[f"{target}_label"][data[f"{target}_mask"] > 0]
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if min(n_pos, n_neg) == 0:
        return 1.0
    return n_neg / n_pos
