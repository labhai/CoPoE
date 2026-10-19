"""Published 8/16/16/8 CoPoE configuration."""

from .model import CCBPoEConfig, RPNS_GROUP_NAMES


SEEDS = (42, 43, 44, 45, 46)
MODALITY_ORDER = ("image", "genomic", "clinical", "biospecimen")
CLASS_NAMES = ("CN", "MCI", "AD")


def paper_config(seed: int, amy_pos_weight: float = 1.0, tau_pos_weight: float = 1.0) -> CCBPoEConfig:
    return CCBPoEConfig(
        in_dim_per_modality=128,
        n_modalities=4,
        scheme="rpns_realloc",
        group_names=RPNS_GROUP_NAMES,
        group_dims=(8, 16, 16, 8),
        graph_mode="no_prior",
        graph_seed=seed,
        encoder_hidden=64,
        dropout=0.15,
        fused_dropout=0.2,
        classifier_hidden=32,
        num_classes=3,
        l1_classifier=1e-5,
        kl_weight=5e-4,
        free_bits=0.02,
        group_prior_weight=0.0,
        anchor_weight=0.0,
        reliability_mode="none",
        residual_mode="mlp",
        residual_hidden=128,
        residual_dropout=0.15,
        residual_scale=0.5,
        enable_biology_heads=True,
        biology_hidden=64,
        biology_head_depth=3,
        biology_dropout=0.2,
        biology_label_smoothing=0.05,
        biology_focal_gamma=1.0,
        biology_use_pos_weight=True,
        biology_amy_pos_weight=amy_pos_weight,
        biology_tau_pos_weight=tau_pos_weight,
        biology_amy_head_lr_mult=2.0,
        biology_tau_head_lr_mult=4.0,
        biology_preset="v2_kitchen_sink",
    )


def attach_paper_heads(model: object) -> None:
    model.attach_specialist_proj([(0, 2), (3, 1)], proj_dim=32)
