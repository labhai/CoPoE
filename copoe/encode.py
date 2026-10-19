"""Export frozen 128-d embeddings from a trained stage-1 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .encoders import build_encoder
from .stage1 import MODALITIES, load_stage1


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modality", required=True, choices=MODALITIES)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--trusted-legacy", action="store_true", help="Allow pickle-based original stage-1 checkpoint")
    return p


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    ids, x, y, _ = load_stage1(args.input, args.modality)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=not args.trusted_legacy)
    if checkpoint.get("modality", args.modality) != args.modality:
        raise ValueError("checkpoint modality does not match --modality")
    if "input_shape" in checkpoint and tuple(checkpoint["input_shape"]) != x.shape[1:]:
        raise ValueError("checkpoint and input feature shapes differ")
    kwargs = dict(checkpoint["model_kwargs"])
    if "input_dim" in kwargs and kwargs["input_dim"] != x.shape[1]:
        raise ValueError("checkpoint input_dim does not match input")
    kwargs.pop("input_dim", None)
    model = build_encoder(args.modality, x.shape[1:], **kwargs)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(args.device).eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), args.batch_size):
            batch = torch.from_numpy(x[start:start + args.batch_size]).to(args.device)
            chunks.append(model.forward_features(batch).cpu().numpy())
    embeddings = np.concatenate(chunks).astype(np.float32)
    if embeddings.shape != (len(y), 128):
        raise ValueError(f"expected [N,128] embedding, got {embeddings.shape}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, ids=ids, y=y, embedding=embeddings)
    print(f"saved {len(y)} embeddings to {args.output}")


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
