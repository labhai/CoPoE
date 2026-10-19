"""Stage-1 modality encoders used before frozen-embedding fusion.

The module structure follows the original stage-1 implementations so their
``model_state`` checkpoints can be loaded without renaming parameters.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn


def _gn_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class Residual3DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.norm1 = nn.GroupNorm(_gn_groups(out_channels), out_channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.norm2 = nn.GroupNorm(_gn_groups(out_channels), out_channels)
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride, bias=False),
                nn.GroupNorm(_gn_groups(out_channels), out_channels),
            )
        else:
            self.skip = nn.Identity()
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.drop(self.norm2(self.conv2(self.act1(self.norm1(self.conv1(x))))))
        return self.act2(x + residual)


class ImageEncoder3D(nn.Module):
    """3D T1 MRI encoder; ``forward_features`` returns the 128-d representation."""

    def __init__(self, embed_dim: int = 128, num_classes: int = 3,
                 base_channels: int = 24, dropout: float = 0.10):
        super().__init__()
        c1, c2, c3, c4 = (base_channels * scale for scale in (1, 2, 4, 8))
        self.stem = nn.Sequential(
            nn.Conv3d(1, c1, 7, 2, 3, bias=False),
            nn.GroupNorm(_gn_groups(c1), c1), nn.GELU(),
            nn.MaxPool3d(3, 2, 1),
        )
        self.stage1 = nn.Sequential(Residual3DBlock(c1, c1, dropout=dropout),
                                    Residual3DBlock(c1, c1, dropout=dropout))
        self.stage2 = nn.Sequential(Residual3DBlock(c1, c2, 2, dropout),
                                    Residual3DBlock(c2, c2, dropout=dropout))
        self.stage3 = nn.Sequential(Residual3DBlock(c2, c3, 2, dropout),
                                    Residual3DBlock(c3, c3, dropout=dropout))
        self.stage4 = nn.Sequential(Residual3DBlock(c3, c4, 2, dropout),
                                    Residual3DBlock(c4, c4, dropout=dropout))
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.embed = nn.Sequential(nn.Flatten(1), nn.LayerNorm(c4), nn.Linear(c4, embed_dim))
        self.classifier = nn.Sequential(nn.LayerNorm(embed_dim), nn.Dropout(dropout),
                                        nn.Linear(embed_dim, num_classes))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for layer in (self.stem, self.stage1, self.stage2, self.stage3, self.stage4,
                      self.pool, self.embed):
            x = layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.10):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.fc2 = nn.Linear(dim * expansion, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.fc2(self.drop(self.act(self.fc1(self.norm(x))))))


class TabularEncoder(nn.Module):
    """Residual MLP for clinical or biospecimen features."""

    def __init__(self, input_dim: int, embed_dim: int = 128, hidden_dim: int = 256,
                 depth: int = 3, dropout: float = 0.10, num_classes: int = 3):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                                        nn.GELU(), nn.Dropout(dropout))
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, expansion=2, dropout=dropout) for _ in range(depth)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.embed = nn.Linear(hidden_dim, embed_dim)
        self.classifier = nn.Sequential(nn.LayerNorm(embed_dim), nn.Dropout(dropout),
                                        nn.Linear(embed_dim, num_classes))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.embed(self.out_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


class GenomicVAEEncoder(nn.Module):
    """Genomic VAE; the deterministic mean is the frozen fusion embedding."""

    def __init__(self, input_dim: int, embed_dim: int = 128,
                 hidden_dims: Sequence[int] = (1024, 512), dropout: float = 0.10,
                 num_classes: int = 3):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must not be empty")
        encoder_layers: List[nn.Module] = []
        prev = input_dim
        for width in hidden_dims:
            encoder_layers.extend((nn.Linear(prev, width), nn.LayerNorm(width),
                                   nn.GELU(), nn.Dropout(dropout)))
            prev = width
        self.encoder = nn.Sequential(*encoder_layers)
        self.mu = nn.Linear(prev, embed_dim)
        self.logvar = nn.Linear(prev, embed_dim)
        decoder_layers: List[nn.Module] = []
        prev = embed_dim
        for width in reversed(tuple(hidden_dims)):
            decoder_layers.extend((nn.Linear(prev, width), nn.LayerNorm(width),
                                   nn.GELU(), nn.Dropout(dropout)))
            prev = width
        decoder_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        self.classifier = nn.Sequential(nn.LayerNorm(embed_dim), nn.Dropout(dropout),
                                        nn.Linear(embed_dim, num_classes))

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor,
                       sample: bool = True) -> torch.Tensor:
        if not sample:
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)[0]

    def forward(self, x: torch.Tensor, sample_latent: Optional[bool] = None) -> Dict[str, torch.Tensor]:
        if sample_latent is None:
            sample_latent = self.training
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, bool(sample_latent))
        return {
            "logits": self.classifier(z), "embedding": mu, "latent": z,
            "recon": self.decode(z), "mu": mu, "logvar": logvar,
        }


def build_encoder(modality: str, input_shape: Sequence[int], **kwargs) -> nn.Module:
    if modality == "image":
        if len(input_shape) != 4 or input_shape[0] != 1:
            raise ValueError("image x must have shape [N,1,D,H,W]")
        return ImageEncoder3D(**kwargs)
    if len(input_shape) != 1:
        raise ValueError("tabular/genomic x must have shape [N,F]")
    if modality == "genomic":
        return GenomicVAEEncoder(input_dim=input_shape[0], **kwargs)
    if modality in ("clinical", "biospecimen"):
        return TabularEncoder(input_dim=input_shape[0], **kwargs)
    raise ValueError(f"unknown modality: {modality}")
