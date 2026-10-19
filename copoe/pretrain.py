"""Train one stage-1 encoder from already preprocessed single-modality arrays."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .encoders import build_encoder
from .stage1 import MODALITIES, load_stage1


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modality", required=True, choices=MODALITIES)
    p.add_argument("--train", required=True, type=Path)
    p.add_argument("--val", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--monitor", choices=("val_loss", "bacc", "auroc"), default="auroc")
    p.add_argument("--balanced-sampler", action="store_true")
    p.add_argument("--class-weighted-loss", action="store_true")
    p.add_argument("--scheduler", choices=("none", "cosine"), default="cosine")
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--base-channels", type=int, default=24)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--genomic-hidden-dims", type=int, nargs="+", default=(1024, 512))
    p.add_argument("--recon-weight", type=float, default=0.2)
    p.add_argument("--kl-beta", type=float, default=0.02)
    return p


def model_kwargs(args):
    modality = args.modality
    kwargs = {"embed_dim": args.embed_dim, "dropout": args.dropout}
    if kwargs["dropout"] is None:
        kwargs["dropout"] = .1
    if modality == "image":
        kwargs["base_channels"] = args.base_channels
    elif modality == "genomic":
        kwargs["hidden_dims"] = tuple(args.genomic_hidden_dims)
    else:
        kwargs["hidden_dim"] = args.hidden_dim or {"clinical": 256, "biospecimen": 192}[modality]
        kwargs["depth"] = args.depth or {"clinical": 3, "biospecimen": 2}[modality]
    return kwargs


def _epoch(model, loader, device, criterion, optimizer=None, modality="clinical",
           recon_weight=0.2, kl_beta=0.02, grad_clip=1.0):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels, predictions, probabilities = [], [], []
    for x, y, missing in loader:
        x, y, missing = x.to(device), y.to(device), missing.to(device)
        with torch.set_grad_enabled(training):
            if modality == "genomic":
                output = model(x, sample_latent=training)
                logits = output["logits"]
                observed = 1.0 - missing
                reconstruction = (((output["recon"] - x) ** 2) * observed).sum() / observed.sum().clamp(min=1)
                mu, logvar = output["mu"], output["logvar"]
                kl = 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar).sum(dim=1).mean() / mu.shape[1]
                loss = criterion(logits, y) + recon_weight * reconstruction + kl_beta * kl if training else criterion(logits, y)
            else:
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        total_loss += float(loss.item()) * len(y)
        labels.extend(y.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        probabilities.append(logits.softmax(dim=1).detach().cpu().numpy())
    try:
        auroc = roc_auc_score(labels, np.concatenate(probabilities), labels=[0, 1, 2], multi_class="ovr", average="macro")
    except ValueError:
        auroc = float("nan")
    return total_loss / len(labels), balanced_accuracy_score(labels, predictions), auroc


def run(args):
    defaults = {"image": (30, 4, 2e-4), "genomic": (40, 16, 2e-4),
                "clinical": (60, 64, 3e-4), "biospecimen": (60, 64, 1e-4)}
    for name, value in zip(("epochs", "batch_size", "lr"), defaults[args.modality]):
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.out}")
    train_ids, train_x, train_y, train_missing = load_stage1(args.train, args.modality)
    val_ids, val_x, val_y, val_missing = load_stage1(args.val, args.modality)
    if set(train_ids.tolist()) & set(val_ids.tolist()):
        raise ValueError("train and validation have overlapping subjects")
    if train_x.shape[1:] != val_x.shape[1:]:
        raise ValueError("train/validation feature shapes differ")
    if len(np.unique(train_y)) < 2:
        raise ValueError("training needs at least two diagnosis classes")
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs, batch size and patience must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    kwargs = model_kwargs(args)
    model = build_encoder(args.modality, train_x.shape[1:], **kwargs).to(device)
    if args.embed_dim != 128:
        raise ValueError("CoPoE fusion requires embed_dim=128")
    class_counts = np.bincount(train_y, minlength=3)
    weights = np.zeros(3, dtype=np.float32)
    weights[class_counts > 0] = len(train_y) / (3.0 * class_counts[class_counts > 0])
    sampler = None
    if args.balanced_sampler:
        sampler = WeightedRandomSampler(torch.from_numpy(weights[train_y]), len(train_y), replacement=True)
    criterion_weights = torch.tensor(weights, device=device) if args.class_weighted_loss else None
    criterion = torch.nn.CrossEntropyLoss(weight=criterion_weights, label_smoothing=args.label_smoothing)

    def loader(x, y, missing, shuffle=False, sampler=None):
        if missing is None:
            missing = np.zeros_like(x, dtype=np.float32) if args.modality == "genomic" else np.zeros((len(y), 1), dtype=np.float32)
        dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(missing))
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler)

    train_loader = loader(train_x, train_y, train_missing, shuffle=sampler is None, sampler=sampler)
    val_loader = loader(val_x, val_y, val_missing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_score = None
    bad_epochs = 0
    rows = []
    args.out.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        if args.scheduler == "cosine":
            if epoch <= args.warmup_epochs:
                factor = epoch / max(1, args.warmup_epochs)
            else:
                progress = (epoch - args.warmup_epochs - 1) / max(1, args.epochs - args.warmup_epochs - 1)
                factor = .5 * (1 + np.cos(np.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * max(.01, factor)
        train_loss, train_bacc, _ = _epoch(model, train_loader, device, criterion, optimizer,
                                          args.modality, args.recon_weight, args.kl_beta, args.grad_clip)
        val_loss, val_bacc, val_auroc = _epoch(model, val_loader, device, criterion, modality=args.modality)
        score = {"val_loss": -val_loss, "bacc": val_bacc, "auroc": val_auroc}[args.monitor]
        if not np.isfinite(score):
            raise ValueError(f"validation {args.monitor} is undefined; check class support")
        rows.append({"epoch": epoch, "train_loss": train_loss, "train_bacc": train_bacc,
                     "val_loss": val_loss, "val_bacc": val_bacc, "val_auroc": val_auroc})
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_bacc={val_bacc:.4f} val_auroc={val_auroc:.4f}", flush=True)
        if best_score is None or score > best_score:
            best_score, bad_epochs = score, 0
            torch.save({"modality": args.modality, "input_shape": tuple(train_x.shape[1:]),
                        "model_kwargs": kwargs, "model_state": model.state_dict(),
                        "best_epoch": epoch}, args.out / "best_encoder.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break
    with (args.out / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    config = {key: value for key, value in vars(args).items() if key not in ("train", "val", "out")}
    config.update({"train_n": len(train_y), "val_n": len(val_y), "best_epoch": torch.load(args.out / "best_encoder.pt", map_location="cpu", weights_only=True)["best_epoch"]})
    (args.out / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
