"""Amyloid and tau readout heads used by the published CoPoE model."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
P_DIM = 16


@dataclass
class BiologyConfig:
    """Architecture and loss settings for the optional PET auxiliary heads."""
    head_hidden: int = 32
    head_depth: int = 2
    head_dropout: float = 0.2
    lambda_amy: float = 0.3
    lambda_tau: float = 0.5
    use_pos_weight: bool = True
    amy_pos_weight: float = -1.0
    tau_pos_weight: float = -1.0
    biospec_downweight: float = 0.5
    label_smoothing: float = 0.05
    focal_gamma: float = 0.0
    tau_head_lr_mult: float = 4.0
    amy_head_lr_mult: float = 1.0
    min_best_epoch: int = 10


class BiologyHeadsV2(nn.Module):
    """Predict amyloid and tau logits from the fused P block (16-d in CoPoE)."""

    def __init__(self, config: BiologyConfig, p_dim: int=P_DIM):
        super().__init__()
        self.config = config

        def make_head(in_dim: int, hidden: int, depth: int, dropout: float) -> nn.Module:
            if depth == 2:
                return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden, 1))
            elif depth == 3:
                return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden, hidden // 2), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden // 2, 1))
            else:
                raise ValueError(f'head_depth must be 2 or 3, got {depth}')
        self.amyloid_head = make_head(p_dim, config.head_hidden, config.head_depth, config.head_dropout)
        self.tau_head = make_head(p_dim, config.head_hidden, config.head_depth, config.head_dropout)

    def forward(self, p_block: torch.Tensor):
        amy_logit = self.amyloid_head(p_block).squeeze(-1)
        tau_logit = self.tau_head(p_block).squeeze(-1)
        return (amy_logit, tau_logit)


def _flexible_bce_loss(logit: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor, pos_weight: float=1.0, label_smoothing: float=0.0, focal_gamma: float=0.0) -> torch.Tensor:
    """Weighted BCE with optional label smoothing and focal modulation."""
    if label_smoothing > 0:
        target_smoothed = target * (1.0 - 2.0 * label_smoothing) + label_smoothing
    else:
        target_smoothed = target
    if pos_weight != 1.0:
        pos_weight_tensor = torch.tensor(pos_weight, device=logit.device)
        bce_per_sample = F.binary_cross_entropy_with_logits(logit, target_smoothed, pos_weight=pos_weight_tensor, reduction='none')
    else:
        bce_per_sample = F.binary_cross_entropy_with_logits(logit, target_smoothed, reduction='none')
    if focal_gamma > 0:
        prob = torch.sigmoid(logit)
        p_t = target * prob + (1 - target) * (1 - prob)
        focal_weight = (1.0 - p_t).clamp(min=1e-07) ** focal_gamma
        bce_per_sample = bce_per_sample * focal_weight
    if sample_weight.sum() > 0:
        return (bce_per_sample * sample_weight).sum() / sample_weight.sum().clamp(min=1e-08)
    else:
        return torch.tensor(0.0, device=logit.device)
