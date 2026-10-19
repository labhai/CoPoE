"""Evaluate a trained CoPoE checkpoint on all observed modality subsets."""

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from .data import load_split
from .metrics import biology_metrics, diagnosis_metrics
from .model import CCBPoEConfig, CCBPoEHead, MODALITY_ORDER, MODALITY_SHORT


def load_model(path: str | Path, device: torch.device, legacy: bool = False) -> CCBPoEHead:
    # Only use legacy=True with a checkpoint you trust; older PyTorch files may contain pickle objects.
    checkpoint = torch.load(path, map_location="cpu", weights_only=not legacy)
    model = CCBPoEHead(CCBPoEConfig(**checkpoint["config"]))
    state = checkpoint["model_state"]
    if any(key.startswith("specialist_proj.") for key in state):
        model.attach_specialist_proj([(0, 2), (3, 1)], proj_dim=32)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def subset_masks() -> list[tuple[int, ...]]:
    result = []
    paper_order = (1, 3, 0, 2)
    for width in range(1, 5):
        for indices in combinations(paper_order, width):
            result.append(tuple(int(i in indices) for i in range(4)))
    return result


@torch.no_grad()
def evaluate_subset(model: CCBPoEHead, data: dict[str, np.ndarray], mask: tuple[int, ...], device: torch.device, batch_size: int = 256) -> dict:
    requested = np.asarray(mask, dtype=np.float32)
    effective = data["mask"] * requested
    valid = effective.sum(axis=1) > 0
    x, observed = data["x"][valid], effective[valid]
    outputs = {"diagnosis": [], "amy": [], "tau": []}
    for start in range(0, len(x), batch_size):
        out = model(
            torch.as_tensor(x[start:start + batch_size], device=device),
            modality_mask=torch.as_tensor(observed[start:start + batch_size], device=device),
            return_dict=True,
        )
        outputs["diagnosis"].append(out["logits"].softmax(dim=-1).cpu().numpy())
        if "amy_logit" in out:
            outputs["amy"].append(out["amy_logit"].sigmoid().cpu().numpy())
            outputs["tau"].append(out["tau_logit"].sigmoid().cpu().numpy())
    prob = np.concatenate(outputs["diagnosis"])
    names = "+".join(MODALITY_SHORT[MODALITY_ORDER[i]] for i in (1, 3, 0, 2) if mask[i])
    row = {"subset": names, "n": int(valid.sum()), **diagnosis_metrics(data["y"][valid], prob)}
    if outputs["amy"]:
        for target in ("amy", "tau"):
            scores = np.concatenate(outputs[target])
            result = biology_metrics(data[f"{target}_label"][valid], data[f"{target}_mask"][valid], scores)
            row.update({f"{target}_{key}": value for key, value in result.items()})
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--legacy-checkpoint", action="store_true", help="Trust and load an original experiment checkpoint")
    args = parser.parse_args()
    device = torch.device(args.device)
    data = load_split(args.data)
    if not (data["mask"].sum(axis=1) == 4).all():
        raise ValueError("The 15-subset benchmark requires the same complete-case subjects for every row")
    model = load_model(args.checkpoint, device, legacy=args.legacy_checkpoint)
    rows = [evaluate_subset(model, data, mask, device, args.batch_size) for mask in subset_masks()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {"all4": rows[-1], "mean_subset_auroc": float(np.nanmean([row["auroc"] for row in rows]))}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
