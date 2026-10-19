"""Train the published CoPoE fusion model from frozen 128-d modality embeddings."""

import argparse
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import SEEDS, attach_paper_heads, paper_config
from .data import class_weights, load_split, loader, positive_weight
from .evaluate import evaluate_subset, subset_masks
from .metrics import biology_metrics, diagnosis_metrics
from .model import CCBPoEHead, compute_ccbpoe_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validation_score(model: CCBPoEHead, data: dict[str, np.ndarray], device: torch.device, paper_selection: bool) -> tuple[float, dict]:
    model.eval()
    diagnosis, amy, tau = [], [], []
    for start in range(0, len(data["y"]), 256):
        out = model(
            torch.as_tensor(data["x"][start:start + 256], device=device),
            torch.as_tensor(data["mask"][start:start + 256], device=device),
            return_dict=True,
        )
        diagnosis.append(out["logits"].softmax(-1).cpu().numpy())
        if model.enable_biology_heads:
            amy.append(out["amy_logit"].sigmoid().cpu().numpy())
            tau.append(out["tau_logit"].sigmoid().cpu().numpy())
    metrics = diagnosis_metrics(data["y"], np.concatenate(diagnosis))
    score = metrics["bacc"]
    if model.enable_biology_heads:
        amy_result = biology_metrics(data["amy_label"], data["amy_mask"], np.concatenate(amy))
        tau_result = biology_metrics(data["tau_label"], data["tau_mask"], np.concatenate(tau))
        metrics.update({"amy_auroc": amy_result["auroc"], "tau_auroc": tau_result["auroc"]})
        if paper_selection:
            score += 0.5 * (amy_result["auroc"] if np.isfinite(amy_result["auroc"]) else 0.5)
            score += 0.5 * (tau_result["auroc"] if np.isfinite(tau_result["auroc"]) else 0.5)
    return score, metrics


def run_seed(seed: int, train_data: dict, val_data: dict, out_dir: Path, device: torch.device, epochs: int, patience: int, batch_size: int, paper_selection: bool, enable_biology: bool = True) -> dict:
    if min(epochs, patience, batch_size) < 1:
        raise ValueError("epochs, patience and batch size must be positive")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out_dir}")
    set_seed(seed)
    cfg = paper_config(seed, positive_weight(train_data, "amy"), positive_weight(train_data, "tau"))
    cfg.enable_biology_heads = enable_biology
    model = CCBPoEHead(cfg)
    attach_paper_heads(model)
    model._biology_config_runtime = model.biology_config
    model._biology_use_pos_weight_runtime = cfg.biology_use_pos_weight
    model._biology_amy_pos_weight_runtime = cfg.biology_amy_pos_weight
    model._biology_tau_pos_weight_runtime = cfg.biology_tau_pos_weight
    model._biology_label_smoothing_runtime = cfg.biology_label_smoothing
    model._biology_focal_gamma_runtime = cfg.biology_focal_gamma
    model = model.to(device)
    amy_params = list(model.biology_heads.amyloid_head.parameters()) if enable_biology else []
    tau_params = list(model.biology_heads.tau_head.parameters()) if enable_biology else []
    biology_ids = {id(param) for param in amy_params + tau_params}
    parameter_groups = [{"params": [p for p in model.parameters() if id(p) not in biology_ids], "lr": 7.5e-5}]
    if enable_biology:
        parameter_groups.extend([{"params": amy_params, "lr": 1.5e-4}, {"params": tau_params, "lr": 3e-4}])
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=5e-4)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=5)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - 5))
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[5])
    batches = loader(train_data, batch_size, seed, train=True)
    sampled_labels = torch.cat([batch[1] for batch in batches]).numpy()
    criterion = nn.CrossEntropyLoss(weight=class_weights(sampled_labels, device), label_smoothing=0.05)
    mask_rng = torch.Generator(device=device).manual_seed(seed)
    best_score, best_epoch, stale = -float("inf"), 0, 0
    history = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        specialist = min(1.0, epoch / 5)
        cascade = 0.1 * min(1.0, epoch / 5)
        kl = cfg.kl_weight * (0.5 + 0.5 * min(1.0, epoch / 5))
        losses = []
        for x, y, mask, anchors, amy_y, amy_mask, tau_y, tau_mask in batches:
            x, y, mask, anchors = (v.to(device) for v in (x, y, mask, anchors))
            amy_y, amy_mask, tau_y, tau_mask = (v.to(device) for v in (amy_y, amy_mask, tau_y, tau_mask))
            optimizer.zero_grad(set_to_none=True)
            loss, _ = compute_ccbpoe_loss(
                model, x, y, mask, anchors, criterion,
                kl_weight=kl, dag_weight=0.0, group_prior_weight=0.0,
                anchor_weight=0.0, free_bits=cfg.free_bits, l1_weight=cfg.l1_classifier,
                n_extra_masks=1, extra_mask_weight=0.25,
                specialist_weight=specialist, specialist_temperature=0.2,
                cascade_weight=cascade, cascade_modalities=("clinical", "genomic"),
                cascade_scheme="rpns_realloc", cascade_group_dims=cfg.group_dims,
                rng=mask_rng, amy_label=amy_y, amy_label_mask=amy_mask,
                tau_label=tau_y, tau_label_mask=tau_mask,
                biology_lambda_amy=0.3 if enable_biology else 0.0,
                biology_lambda_tau=0.7 if enable_biology else 0.0,
                biology_biospec_downweight=0.5,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        score, metrics = validation_score(model, val_data, device, paper_selection)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "score": score, **metrics})
        print(f"seed={seed} epoch={epoch:02d} loss={np.mean(losses):.4f} val_bacc={metrics['bacc']:.4f} selection={score:.4f}", flush=True)
        min_best_epoch = min(10, epochs)
        if epoch >= min_best_epoch and score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({"config": asdict(cfg), "model_state": model.state_dict(), "summary": {"seed": seed, "epoch": epoch, "selection_score": score}}, out_dir / "best.pt")
        elif epoch >= min_best_epoch:
            stale += 1
        if stale >= patience:
            break
    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    return {"seed": seed, "best_epoch": best_epoch, "best_score": best_score}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--selection", choices=("paper", "bacc"), default="paper",
        help="'paper' preserves the original full-mask BAcc + 0.5 amy AUROC + 0.5 tau AUROC criterion; "
             "'bacc' uses full-mask BAcc. Neither averages over 15 subsets.",
    )
    parser.add_argument("--diagnosis-only", action="store_true", help="Disable PET auxiliary heads when PET labels are unavailable")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train_data, val_data = load_split(args.train), load_split(args.val)
    if args.selection == "paper" and args.diagnosis_only:
        parser.error("--selection paper requires PET heads; use --selection bacc with --diagnosis-only")
    if not args.diagnosis_only:
        for target in ("amy", "tau"):
            for split_name, split in (("train", train_data), ("val", val_data)):
                observed = split[f"{target}_label"][split[f"{target}_mask"] > 0]
                if set(observed.tolist()) != {0, 1}:
                    parser.error(f"{split_name}: {target} needs both PET classes; use --selection bacc --diagnosis-only")
    if not (val_data["mask"].sum(axis=1) == 4).all():
        raise ValueError("Validation must contain complete-case subjects")
    test_data = load_split(args.test) if args.test else None
    if test_data is not None and not (test_data["mask"].sum(axis=1) == 4).all():
        raise ValueError("15-subset testing must contain complete-case subjects")
    device = torch.device(args.device)
    all_results = []
    for seed in args.seeds:
        out_dir = args.out / f"seed{seed}"
        result = run_seed(seed, train_data, val_data, out_dir, device, args.epochs, args.patience,
                          args.batch_size, args.selection == "paper", enable_biology=not args.diagnosis_only)
        if test_data is not None:
            from .evaluate import load_model
            model = load_model(out_dir / "best.pt", device)
            rows = [evaluate_subset(model, test_data, mask, device) for mask in subset_masks()]
            with (out_dir / "test_subsets.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            result["test_all4"] = rows[-1]
            result["test_mean_auroc_15"] = float(np.nanmean([row["auroc"] for row in rows]))
        all_results.append(result)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
