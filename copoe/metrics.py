"""Diagnostic and PET metrics used for the release evaluator."""

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score


def diagnosis_metrics(y: np.ndarray, prob: np.ndarray, bins: int = 15) -> dict[str, float]:
    pred = prob.argmax(axis=1)
    conf = prob.max(axis=1)
    correct = (pred == y).astype(np.float64)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        selected = (conf >= edges[i]) & (conf < edges[i + 1] if i < bins - 1 else conf <= edges[i + 1])
        if selected.any():
            ece += selected.mean() * abs(correct[selected].mean() - conf[selected].mean())
    one_hot = np.eye(3)[y]
    clipped = np.clip(prob[np.arange(len(y)), y], 1e-12, 1.0)
    return {
        "bacc": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(y, prob, multi_class="ovr", average="macro")) if len(np.unique(y)) == 3 else float("nan"),
        "ece": float(ece),
        "brier": float(np.mean(np.sum((prob - one_hot) ** 2, axis=1))),
        "nll": float(-np.log(clipped).mean()),
    }


def biology_metrics(y: np.ndarray, available: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    valid = available > 0
    if valid.sum() == 0 or len(np.unique(y[valid])) < 2:
        return {"n": int(valid.sum()), "auroc": float("nan"), "auprc": float("nan")}
    return {
        "n": int(valid.sum()),
        "auroc": float(roc_auc_score(y[valid], prob[valid])),
        "auprc": float(average_precision_score(y[valid], prob[valid])),
    }
