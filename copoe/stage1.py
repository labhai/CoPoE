"""Validated, preprocessed single-modality arrays for encoder training/export."""

from __future__ import annotations

from pathlib import Path

import numpy as np


MODALITIES = ("image", "genomic", "clinical", "biospecimen")


def load_stage1(path: str | Path, modality: str):
    if modality not in MODALITIES:
        raise ValueError(f"modality must be one of {MODALITIES}")
    path = Path(path)
    if path.is_dir():
        required = {"ids", "x", "y"}
        present = {name for name in (*required, "missing_mask") if (path / f"{name}.npy").exists()}
        if not required.issubset(present):
            raise ValueError(f"{path}: missing {sorted(required - present)} .npy arrays")
        arrays = {name: np.load(path / f"{name}.npy", mmap_mode="c", allow_pickle=False)
                  for name in present}
        if arrays["x"].dtype != np.float32:
            raise ValueError(f"{path}: memory-mapped x.npy must be float32")
    else:
        with np.load(path, allow_pickle=False) as data:
            required = {"ids", "x", "y"}
            if not required.issubset(data.files):
                raise ValueError(f"{path}: missing {sorted(required - set(data.files))}")
            arrays = {name: data[name] for name in (*required, "missing_mask") if name in data.files}
    ids = np.asarray(arrays["ids"])
    x = np.asarray(arrays["x"], dtype=np.float32)
    y = np.asarray(arrays["y"])
    missing = np.asarray(arrays["missing_mask"], dtype=np.float32) if "missing_mask" in arrays else None
    if ids.dtype.kind not in "US" or ids.ndim != 1:
        raise ValueError("ids must be a 1-D Unicode/string array, not object/pickle")
    # Normalize byte strings before comparing splits: b"subject" and "subject"
    # refer to the same participant and must not bypass the leakage check.
    if ids.dtype.kind == "S":
        ids = np.char.decode(ids, "utf-8")
    if any(not identifier.strip() for identifier in ids.tolist()):
        raise ValueError(f"{path}: empty subject IDs")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError(f"{path}: duplicate subject IDs")
    if y.ndim != 1 or x.ndim == 0 or x.shape[0] != len(y) or len(y) != len(ids) or len(y) == 0:
        raise ValueError(f"{path}: inconsistent or empty subject axis")
    if modality == "image":
        if x.ndim != 5 or x.shape[1] != 1 or min(x.shape[2:]) < 16:
            raise ValueError("image x must have shape [N,1,D,H,W] with D/H/W >= 16")
    elif x.ndim != 2:
        raise ValueError(f"{modality} x must have shape [N,F]")
    if min(x.shape[1:]) < 1:
        raise ValueError(f"{path}: x has an empty feature dimension")
    step = max(1, 16_000_000 // int(np.prod(x.shape[1:])))
    for start in range(0, len(x), step):
        if not np.isfinite(x[start:start + step]).all():
            raise ValueError(f"{path}: x contains NaN/inf; impute using train-fitted rules first")
    if not np.isin(y, (0, 1, 2)).all():
        raise ValueError(f"{path}: y must contain only 0=CN, 1=MCI, 2=AD")
    y = y.astype(np.int64)
    if missing is not None:
        if modality != "genomic" or missing.shape != x.shape or not np.isin(missing, (0, 1)).all():
            raise ValueError("missing_mask is genomic-only, binary, and must match x")
    return ids, x, y, missing
