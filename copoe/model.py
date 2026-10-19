"""CoPoE disease-coordinate Gaussian experts, fusion, and losses."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .biology import BiologyConfig, BiologyHeadsV2, _flexible_bce_loss

NUM_CLASSES = 3
CLASS_NAMES = ("CN", "MCI", "AD")
MODALITY_ORDER = ("image", "genomic", "clinical", "biospecimen")
MODALITY_SHORT = {"image": "I", "genomic": "G", "clinical": "C", "biospecimen": "B"}
RPNS_GROUP_NAMES = ("genetic_risk", "biofluid_pathology", "neurodegeneration", "clinical_stage")
SCHEME_TO_GROUP_NAMES = {"rpns": RPNS_GROUP_NAMES, "rpns_realloc": RPNS_GROUP_NAMES, "flat": ("flat_latent",)}
SCHEME_TO_DEFAULT_DIMS = {"rpns": (8, 16, 12, 8), "rpns_realloc": (8, 16, 16, 8), "flat": (48,)}
MODALITY_GROUP_PRIOR_BY_SCHEME = {
    "rpns": {"genomic": ("genetic_risk",), "biospecimen": ("biofluid_pathology",),
             "image": ("neurodegeneration",), "clinical": ("clinical_stage",)},
    "flat": {},
}
MODALITY_GROUP_PRIOR_BY_SCHEME["rpns_realloc"] = MODALITY_GROUP_PRIOR_BY_SCHEME["rpns"]


def _cascade_prior_table(scheme: str) -> Dict[str, 'torch.Tensor']:
    import torch as _torch
    if scheme in ('rpns', 'rpns_realloc'):
        return {'image': _torch.tensor([0.05, 0.15, 0.7, 0.1]), 'biospecimen': _torch.tensor([0.05, 0.8, 0.1, 0.05]), 'clinical': _torch.tensor([0.05, 0.2, 0.3, 0.45]), 'genomic': _torch.tensor([0.4, 0.4, 0.1, 0.1])}
    elif scheme in ('rpnsx', 'rpnsx_realloc'):
        return {'image': _torch.tensor([0.04, 0.13, 0.65, 0.08, 0.1]), 'biospecimen': _torch.tensor([0.04, 0.75, 0.08, 0.04, 0.09]), 'clinical': _torch.tensor([0.05, 0.15, 0.2, 0.3, 0.3]), 'genomic': _torch.tensor([0.3, 0.3, 0.05, 0.05, 0.3])}
    else:
        return {}


def _cascade_kl_loss_v2(out, modality_mask, scheme, target_modalities, group_dims, prior_module=None):
    """Regularize observed-modality precision profiles toward fixed targets."""
    import torch as _torch, torch.nn.functional as _F
    pi_table_init = _cascade_prior_table(scheme)
    if not pi_table_init or not target_modalities:
        return out['logits'].new_zeros(())
    logvar_mod = out['logvar_mod']
    offsets, c = ([], 0)
    for d in group_dims:
        offsets.append((c, c + d))
        c += d
    mod_to_idx = {'image': 0, 'genomic': 1, 'clinical': 2, 'biospecimen': 3}
    losses = []
    for m_name in target_modalities:
        m_name = m_name.strip().lower()
        if m_name not in mod_to_idx or m_name not in pi_table_init:
            continue
        m_idx = mod_to_idx[m_name]
        obs = modality_mask[:, m_idx] > 0.5
        if obs.sum() < 1:
            continue
        block_prec = []
        for s, e in offsets:
            tau = _torch.exp(-logvar_mod[obs, m_idx, s:e]).mean(dim=-1)
            block_prec.append(tau)
        block_prec = _torch.stack(block_prec, dim=-1)
        P = block_prec / (block_prec.sum(dim=-1, keepdim=True) + 1e-08)
        if prior_module is None or prior_module.mode == 'fixed':
            pi = pi_table_init[m_name].to(P.device)
            pi = pi / pi.sum()
            pi_b = pi.unsqueeze(0)
        elif prior_module.mode == 'cohort':
            pi = prior_module.get_prior(m_name, modality_mask)
            pi_b = pi.unsqueeze(0)
        elif prior_module.mode == 'sample':
            obs_mask = modality_mask[obs]
            pi_b = prior_module.get_prior(m_name, obs_mask)
        else:
            pi = pi_table_init[m_name].to(P.device)
            pi_b = (pi / pi.sum()).unsqueeze(0)
        kl = (P * (P.clamp(1e-08).log() - pi_b.clamp(1e-08).log())).sum(dim=-1).mean()
        losses.append(kl)
    if not losses:
        return out['logits'].new_zeros(())
    return _torch.stack(losses).mean()


def normalize_scheme(scheme: str) -> str:
    scheme = str(scheme).lower().strip()
    if scheme not in SCHEME_TO_GROUP_NAMES:
        raise ValueError(f'Unknown CCB-PoE scheme: {scheme}. Choose from {sorted(SCHEME_TO_GROUP_NAMES)}')
    return scheme


def default_group_dims_for_scheme(scheme: str) -> Tuple[int, ...]:
    return tuple(SCHEME_TO_DEFAULT_DIMS[normalize_scheme(scheme)])


def group_names_for_scheme(scheme: str) -> Tuple[str, ...]:
    return tuple(SCHEME_TO_GROUP_NAMES[normalize_scheme(scheme)])


def build_group_parents(n_groups: int, mode: str='canonical', seed: int=42) -> Tuple[Tuple[int, ...], ...]:
    """Return parent sets for the concept graph.

    RPNS canonical graph:
      R -> P -> N -> S, with later blocks allowed to use all upstream blocks.

    RATS canonical graph:
      R -> A -> T -> N -> S, with later blocks allowed to use all upstream blocks.

    This is not claimed as causal identification. It is a disease-cascade prior.
    """
    mode = str(mode).lower()
    if n_groups == 1:
        return (tuple(),)
    if n_groups not in (4, 5):
        raise ValueError('CCB-PoE supports 1-group flat, 4-group RPNS, or 5-group RATS concepts.')
    if mode == 'canonical':
        if n_groups == 4:
            return (tuple(), (0,), (0, 1), (0, 1, 2))
        return (tuple(), (0,), (0, 1), (0, 1, 2), (0, 1, 2, 3))
    if mode == 'chain':
        return tuple((tuple() if k == 0 else (k - 1,) for k in range(n_groups)))
    if mode == 'no_prior':
        return tuple((tuple() for _ in range(n_groups)))
    if mode == 'reversed':
        return tuple((tuple(range(k + 1, n_groups)) for k in range(n_groups)))
    if mode == 'random':
        rng = np.random.default_rng(int(seed))
        parents: List[Tuple[int, ...]] = [tuple()]
        for k in range(1, n_groups):
            cand = [j for j in range(k) if rng.random() < 0.5]
            if len(cand) == 0:
                cand = [int(rng.integers(0, k))]
            parents.append(tuple(cand))
        return tuple(parents)
    raise ValueError(f'Unknown graph mode: {mode}')


def _group_slices(group_dims: Sequence[int]) -> Tuple[slice, ...]:
    out: List[slice] = []
    s = 0
    for d in group_dims:
        out.append(slice(s, s + int(d)))
        s += int(d)
    return tuple(out)


class _MLP(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, hidden: int=64, dropout: float=0.15):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModalityConceptEncoder(nn.Module):
    """Map one modality embedding to a Gaussian belief over all concept dims."""

    def __init__(self, in_dim: int, concept_dim: int, hidden: int=64, dropout: float=0.15, logvar_clamp: Tuple[float, float]=(-4.0, 2.0)):
        super().__init__()
        self.mu = _MLP(in_dim, concept_dim, hidden=hidden, dropout=dropout)
        self.logvar = _MLP(in_dim, concept_dim, hidden=hidden, dropout=dropout)
        self.logvar_clamp = logvar_clamp

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu(z)
        logvar = self.logvar(z).clamp(*self.logvar_clamp)
        return (mu, logvar)


class CascadeGaussianPrior(nn.Module):
    """Linear Gaussian concept prior induced by an R+AT(N)+S cascade graph."""

    def __init__(self, group_dims: Sequence[int], group_parents: Sequence[Sequence[int]], learn_edges: bool=True, init_logvar: float=0.0, logvar_clamp: Tuple[float, float]=(-2.0, 2.0)):
        super().__init__()
        self.group_dims = tuple((int(d) for d in group_dims))
        self.group_parents = tuple((tuple((int(j) for j in p)) for p in group_parents))
        self.n_groups = len(self.group_dims)
        self.concept_dim = int(sum(self.group_dims))
        self.group_slices = _group_slices(self.group_dims)
        self.logvar_clamp = logvar_clamp
        self.edges = nn.ParameterDict()
        for k, parents in enumerate(self.group_parents):
            for j in parents:
                w = torch.zeros(self.group_dims[k], self.group_dims[j])
                if learn_edges:
                    nn.init.normal_(w, mean=0.0, std=0.05)
                self.edges[f'e_{j}_to_{k}'] = nn.Parameter(w, requires_grad=learn_edges)
        self.logvar_raw = nn.Parameter(torch.full((self.concept_dim,), float(init_logvar)))

    def prior_mean_from_point_estimate(self, c_point: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(c_point)
        for k, parents in enumerate(self.group_parents):
            if len(parents) == 0:
                continue
            sl_k = self.group_slices[k]
            contrib = None
            for j in parents:
                sl_j = self.group_slices[j]
                A = self.edges[f'e_{j}_to_{k}']
                term = c_point[:, sl_j] @ A.t()
                contrib = term if contrib is None else contrib + term
            out[:, sl_k] = contrib
        return out

    def logvar_vector(self) -> torch.Tensor:
        return self.logvar_raw.clamp(*self.logvar_clamp)


class ConceptAnchorHead(nn.Module):
    """Optional scalar anchor predictor for each concept block.

    If a concept anchor CSV is supplied, each group receives one scalar target
    such as normalized Aβ ratio, p-tau, hippocampal volume summary, MMSE, etc.
    Missing targets should be NaN and are skipped by the loss.
    """

    def __init__(self, group_dims: Sequence[int], hidden: int=16):
        super().__init__()
        self.group_dims = tuple((int(d) for d in group_dims))
        self.group_slices = _group_slices(self.group_dims)
        self.heads = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, 1)) for d in self.group_dims])

    def forward(self, concept_mean: torch.Tensor) -> torch.Tensor:
        preds = []
        for g, sl in enumerate(self.group_slices):
            preds.append(self.heads[g](concept_mean[:, sl]).squeeze(-1))
        return torch.stack(preds, dim=1)


def _parse_specialist_pairs(spec: str) -> List[Tuple[int, int]]:
    """Parse 'I:N,B:P' -> [(0, 2), (3, 1)] using MODALITY_ORDER and RPNS_GROUP_NAMES."""
    if not spec or spec.strip().lower() == 'none':
        return []
    mod_short = {'I': 0, 'G': 1, 'C': 2, 'B': 3}
    blk_short = {'R': 0, 'P': 1, 'N': 2, 'S': 3}
    out = []
    for tok in spec.split(','):
        tok = tok.strip()
        if not tok:
            continue
        m_, b_ = tok.split(':')
        out.append((mod_short[m_.strip().upper()], blk_short[b_.strip().upper()]))
    return out


class SpecialistProjections(nn.Module):
    """Two small projection heads (one per pair) mapping per-modality block-mu
    to a shared critic space for InfoNCE."""

    def __init__(self, group_dims: Sequence[int], pairs: List[Tuple[int, int]], proj_dim: int=32):
        super().__init__()
        self.pairs = list(pairs)
        self.group_dims = list(group_dims)
        offsets, c = ([], 0)
        for d in self.group_dims:
            offsets.append(c)
            c += d
        self.offsets = offsets
        self.proj_dim = proj_dim
        self.heads = nn.ModuleList()
        for m_idx, b_idx in self.pairs:
            d = self.group_dims[b_idx]
            self.heads.append(nn.Sequential(nn.Linear(d, proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim)))

    def slice_block(self, mu_mod_or_fuse: torch.Tensor, b_idx: int) -> torch.Tensor:
        """Slice [..., concept_dim] -> [..., group_dims[b_idx]] for block b_idx."""
        off, d = (self.offsets[b_idx], self.group_dims[b_idx])
        return mu_mod_or_fuse[..., off:off + d]


def specialist_infonce_loss(out: Dict[str, torch.Tensor], proj: 'SpecialistProjections', modality_mask: torch.Tensor, temperature: float=0.2) -> torch.Tensor:
    """For each (m_idx, b_idx) pair, an InfoNCE between:
        anchor = projected per-modality mu of modality m_idx in block b_idx
        positive = projected fused mu in block b_idx (same sample)
        negatives = projected fused mu in block b_idx of other samples in batch
    Only over samples where modality m is observed.
    """
    if not proj.pairs:
        return out['logits'].new_zeros(())
    mu_mod = out['mu_mod']
    mu_fuse = out['concept_mean']
    losses = []
    for k, (m_idx, b_idx) in enumerate(proj.pairs):
        obs = modality_mask[:, m_idx] > 0.5
        if obs.sum() < 2:
            continue
        head = proj.heads[k]
        a = head(proj.slice_block(mu_mod[obs, m_idx], b_idx))
        f = head(proj.slice_block(mu_fuse[obs], b_idx))
        a = F.normalize(a, dim=-1)
        f = F.normalize(f, dim=-1)
        sim = a @ f.t() / max(1e-06, temperature)
        target = torch.arange(a.size(0), device=a.device)
        losses.append(F.cross_entropy(sim, target))
    if not losses:
        return out['logits'].new_zeros(())
    return torch.stack(losses).mean()


@dataclass
class CCBPoEConfig:
    in_dim_per_modality: int = 128
    n_modalities: int = 4
    scheme: str = 'rpns'
    group_names: Tuple[str, ...] = RPNS_GROUP_NAMES
    group_dims: Tuple[int, ...] = (8, 16, 12, 8)
    graph_mode: str = 'canonical'
    graph_seed: int = 42
    encoder_hidden: int = 64
    dropout: float = 0.15
    fused_dropout: float = 0.2
    classifier_hidden: int = 32
    num_classes: int = NUM_CLASSES
    l1_classifier: float = 1e-05
    kl_weight: float = 0.02
    dag_consistency_weight: float = 0.0
    group_prior_weight: float = 0.05
    group_prior_floor: float = 0.02
    group_align_mode: str = 'kl'
    group_margin_delta: float = 0.1
    group_contrib_floor: float = 0.02
    group_owner_min_rho: float = 0.35
    clinical_prior_mode: str = 'stage'
    anchor_weight: float = 0.0
    free_bits: float = 0.05
    dag_max_scale: float = 1.0
    reliability_mode: str = 'none'
    reliability_target: float = 0.9
    reliability_min: float = 0.05
    reliability_hidden: int = 32
    reliability_reg_weight: float = 0.0
    fusion_mode: str = 'poe'
    concept_logit_weight: float = 1.0
    residual_mode: str = 'none'
    residual_hidden: int = 128
    residual_dropout: float = 0.15
    residual_scale: float = 0.0
    residual_learnable: bool = False
    prior_sample: bool = False
    learn_prior_edges: bool = True
    enable_biology_heads: bool = False
    biology_hidden: int = 32
    biology_head_depth: int = 2
    biology_dropout: float = 0.2
    biology_label_smoothing: float = 0.0
    biology_focal_gamma: float = 0.0
    biology_use_pos_weight: bool = False
    biology_amy_pos_weight: float = 1.0
    biology_tau_pos_weight: float = 1.0
    biology_orthogonality_weight: float = 0.0
    biology_amy_head_lr_mult: float = 1.0
    biology_tau_head_lr_mult: float = 1.0
    biology_preset: str = 'manual'


def _safe_logit(p: float, eps: float=0.0001) -> float:
    p = float(max(eps, min(1.0 - eps, p)))
    return math.log(p / (1.0 - p))


def _masked_concat_input(x: torch.Tensor, modality_mask: torch.Tensor, in_dim_per_modality: int) -> torch.Tensor:
    """Zero out modality blocks according to an availability mask."""
    B = x.shape[0]
    pieces = []
    for m in range(modality_mask.shape[1]):
        s = m * in_dim_per_modality
        e = (m + 1) * in_dim_per_modality
        pieces.append(x[:, s:e] * modality_mask[:, m].view(B, 1).to(x.dtype))
    return torch.cat(pieces, dim=1)


class CCBPoEHead(nn.Module):
    """Cascade-guided concept bottleneck + masked Gaussian PoE fusion.

    Optional extensions:
      1) reliability-tempered PoE, learning alpha_m for each observed modality;
      2) residual direct evidence head over masked Stage-1 embeddings.
    """

    def __init__(self, cfg: Optional[CCBPoEConfig]=None, **overrides):
        super().__init__()
        if cfg is None:
            cfg = CCBPoEConfig(**overrides)
        elif overrides:
            cfg = CCBPoEConfig(**{**asdict(cfg), **overrides})
        cfg.scheme = normalize_scheme(getattr(cfg, 'scheme', 'rpns'))
        if not cfg.group_names:
            cfg.group_names = group_names_for_scheme(cfg.scheme)
        self.cfg = cfg
        self.group_names = tuple(cfg.group_names)
        self.group_dims = tuple((int(d) for d in cfg.group_dims))
        self.group_slices = _group_slices(self.group_dims)
        self.concept_dim = int(sum(self.group_dims))
        self.n_modalities = int(cfg.n_modalities)
        self.group_parents = build_group_parents(n_groups=len(self.group_dims), mode=cfg.graph_mode, seed=cfg.graph_seed)
        self.encoders = nn.ModuleList([ModalityConceptEncoder(in_dim=cfg.in_dim_per_modality, concept_dim=self.concept_dim, hidden=cfg.encoder_hidden, dropout=cfg.dropout) for _ in range(self.n_modalities)])
        self.prior = CascadeGaussianPrior(group_dims=self.group_dims, group_parents=self.group_parents, learn_edges=cfg.learn_prior_edges)
        self.concept_ln = nn.LayerNorm(self.concept_dim)
        self.concept_dropout = nn.Dropout(cfg.fused_dropout)
        if cfg.classifier_hidden and cfg.classifier_hidden > 0:
            self.classifier = nn.Sequential(nn.Linear(self.concept_dim, cfg.classifier_hidden), nn.GELU(), nn.Dropout(cfg.fused_dropout), nn.Linear(cfg.classifier_hidden, cfg.num_classes))
        else:
            self.classifier = nn.Linear(self.concept_dim, cfg.num_classes)
        self.enable_biology_heads = bool(getattr(cfg, 'enable_biology_heads', False))
        self.biology_block_index = self.group_names.index('biofluid_pathology') if 'biofluid_pathology' in self.group_names else None
        self.biology_slice: Optional[slice] = None
        if self.enable_biology_heads:
            if self.biology_block_index is not None:
                self.biology_slice = self.group_slices[self.biology_block_index]
                p_dim = int(self.group_dims[self.biology_block_index])
            elif str(getattr(cfg, 'scheme', '')).lower().strip() == 'flat':
                self.biology_slice = slice(0, self.concept_dim)
                p_dim = int(self.concept_dim)
            else:
                raise ValueError("Biology heads require a 'biofluid_pathology' concept block, except scheme=flat where they attach to the full latent.")
            biology_cfg = BiologyConfig(head_hidden=int(getattr(cfg, 'biology_hidden', 32)), head_depth=int(getattr(cfg, 'biology_head_depth', 2)), head_dropout=float(getattr(cfg, 'biology_dropout', 0.2)), label_smoothing=float(getattr(cfg, 'biology_label_smoothing', 0.0)), focal_gamma=float(getattr(cfg, 'biology_focal_gamma', 0.0)), use_pos_weight=bool(getattr(cfg, 'biology_use_pos_weight', False)), amy_pos_weight=float(getattr(cfg, 'biology_amy_pos_weight', 1.0)), tau_pos_weight=float(getattr(cfg, 'biology_tau_pos_weight', 1.0)), amy_head_lr_mult=float(getattr(cfg, 'biology_amy_head_lr_mult', 1.0)), tau_head_lr_mult=float(getattr(cfg, 'biology_tau_head_lr_mult', 1.0)))
            self.biology_config = biology_cfg
            self.biology_heads = BiologyHeadsV2(biology_cfg, p_dim=p_dim)
        else:
            self.biology_config = None
            self.biology_heads = None
        self.mod_scale = nn.Parameter(torch.zeros(self.n_modalities))
        self.anchor_head = ConceptAnchorHead(self.group_dims)
        self.specialist_proj: Optional[SpecialistProjections] = None
        self.reliability_mode = str(getattr(cfg, 'reliability_mode', 'none')).lower().strip()
        if self.reliability_mode not in {'none', 'global', 'sample', 'both'}:
            raise ValueError('reliability_mode must be one of: none, global, sample, both')
        self.reliability_min = float(max(0.0, min(0.95, getattr(cfg, 'reliability_min', 0.05))))
        rel_target = float(max(self.reliability_min + 0.0001, min(0.999, getattr(cfg, 'reliability_target', 0.9))))
        raw_target = (rel_target - self.reliability_min) / max(1e-06, 1.0 - self.reliability_min)
        base_logit = _safe_logit(raw_target)
        self.register_buffer('reliability_base_logit', torch.full((self.n_modalities,), float(base_logit)))
        if self.reliability_mode in {'global', 'both'}:
            self.reliability_logits = nn.Parameter(torch.full((self.n_modalities,), float(base_logit)))
        else:
            self.register_parameter('reliability_logits', None)
        if self.reliability_mode in {'sample', 'both'}:
            gates = []
            for _ in range(self.n_modalities):
                gate = nn.Sequential(nn.LayerNorm(2 * self.concept_dim), nn.Linear(2 * self.concept_dim, int(getattr(cfg, 'reliability_hidden', 32))), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(int(getattr(cfg, 'reliability_hidden', 32)), 1))
                nn.init.zeros_(gate[-1].weight)
                nn.init.zeros_(gate[-1].bias)
                gates.append(gate)
            self.reliability_gates = nn.ModuleList(gates)
        else:
            self.reliability_gates = nn.ModuleList()
        self.residual_mode = str(getattr(cfg, 'residual_mode', 'none')).lower().strip()
        if self.residual_mode not in {'none', 'linear', 'mlp'}:
            raise ValueError('residual_mode must be one of: none, linear, mlp')
        residual_in = self.n_modalities * int(cfg.in_dim_per_modality)
        if self.residual_mode == 'linear':
            self.residual_head = nn.Sequential(nn.LayerNorm(residual_in), nn.Dropout(float(getattr(cfg, 'residual_dropout', 0.15))), nn.Linear(residual_in, cfg.num_classes))
        elif self.residual_mode == 'mlp':
            self.residual_head = nn.Sequential(nn.LayerNorm(residual_in), nn.Linear(residual_in, int(getattr(cfg, 'residual_hidden', 128))), nn.GELU(), nn.Dropout(float(getattr(cfg, 'residual_dropout', 0.15))), nn.Linear(int(getattr(cfg, 'residual_hidden', 128)), cfg.num_classes))
        else:
            self.residual_head = None
        if bool(getattr(cfg, 'residual_learnable', False)):
            self.residual_scale_logit = nn.Parameter(torch.tensor(_safe_logit(float(getattr(cfg, 'residual_scale', 0.0)))))
        else:
            self.register_buffer('residual_scale_fixed', torch.tensor(float(getattr(cfg, 'residual_scale', 0.0))))
            self.register_parameter('residual_scale_logit', None)
        self.register_buffer('modality_group_targets', self._make_modality_group_targets())
        self.register_buffer('modality_group_primary_mask', self._make_modality_group_primary_mask())

    def _make_modality_group_targets(self) -> torch.Tensor:
        floor = max(0.0, float(getattr(self.cfg, 'group_prior_floor', 0.02)))
        targets = torch.full((self.n_modalities, len(self.group_dims)), floor, dtype=torch.float32)
        name_to_idx = {g: i for i, g in enumerate(self.group_names)}
        clinical_mode = str(getattr(self.cfg, 'clinical_prior_mode', 'stage')).lower()
        for m_idx, m_name in enumerate(MODALITY_ORDER):
            primary = MODALITY_GROUP_PRIOR_BY_SCHEME.get(self.cfg.scheme, {}).get(m_name, tuple())
            if m_name == 'clinical':
                if clinical_mode == 'stage':
                    primary = ('clinical_stage',)
                elif clinical_mode == 'stage_neuro':
                    primary = ('clinical_stage', 'neurodegeneration')
                elif clinical_mode == 'uniform':
                    primary = self.group_names
                else:
                    raise ValueError(f'Unknown clinical_prior_mode: {clinical_mode}')
            for g_name in primary:
                if g_name in name_to_idx:
                    targets[m_idx, name_to_idx[g_name]] = 1.0
            targets[m_idx] = targets[m_idx] / targets[m_idx].sum()
        return targets

    def _make_modality_group_primary_mask(self) -> torch.Tensor:
        """Binary M x G mask of primary modality anchors.

        This is derived from the soft target distribution and is used only by
        relaxed anchor losses such as margin/contribution-margin. It does not
        make the forward path a hard one-to-one mapping.
        """
        target = self.modality_group_targets.detach().clone()
        maxv = target.max(dim=1, keepdim=True).values
        return (target >= maxv - 1e-06).to(torch.float32)

    def _split_modalities(self, x: torch.Tensor) -> List[torch.Tensor]:
        D = int(self.cfg.in_dim_per_modality)
        return [x[:, m * D:(m + 1) * D] for m in range(self.n_modalities)]

    def _encode_all(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        parts = self._split_modalities(x)
        mus, logvars = ([], [])
        for m in range(self.n_modalities):
            mu_m, logvar_m = self.encoders[m](parts[m])
            logvar_m = logvar_m - self.mod_scale[m]
            if str(getattr(self.cfg, 'fusion_mode', 'poe')).lower().strip() in {'unit_variance', 'deterministic'}:
                logvar_m = torch.zeros_like(logvar_m)
            mus.append(mu_m)
            logvars.append(logvar_m)
        return (torch.stack(mus, dim=1), torch.stack(logvars, dim=1))

    @staticmethod
    def _poe_from_nat(precision_sum: torch.Tensor, eta_sum: torch.Tensor, eps: float=1e-06):
        var = 1.0 / (precision_sum + eps)
        mu = var * eta_sum
        logvar = torch.log(var + eps)
        return (mu, logvar)

    @staticmethod
    def kl_diag_gaussians_per_dim(mu_q: torch.Tensor, logvar_q: torch.Tensor, mu_p: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        return 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p).pow(2)) / (var_p + 1e-08) - 1.0)

    def _compute_reliability_alpha(self, mu_mod: torch.Tensor, logvar_mod: torch.Tensor) -> torch.Tensor:
        B, M, _ = mu_mod.shape
        if self.reliability_mode == 'none':
            return torch.ones(B, M, device=mu_mod.device, dtype=mu_mod.dtype)
        if self.reliability_mode in {'global', 'both'}:
            logits = self.reliability_logits.to(device=mu_mod.device, dtype=mu_mod.dtype).unsqueeze(0).expand(B, -1)
        else:
            logits = self.reliability_base_logit.to(device=mu_mod.device, dtype=mu_mod.dtype).unsqueeze(0).expand(B, -1).clone()
        if self.reliability_mode in {'sample', 'both'}:
            gate_inputs = torch.cat([mu_mod, logvar_mod], dim=-1)
            gate_logits = []
            for m in range(M):
                gate_logits.append(self.reliability_gates[m](gate_inputs[:, m]).squeeze(-1))
            logits = logits + torch.stack(gate_logits, dim=1)
        alpha01 = torch.sigmoid(logits)
        return self.reliability_min + (1.0 - self.reliability_min) * alpha01

    def reliability_regularization_loss(self, out: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.reliability_mode == 'none' or float(getattr(self.cfg, 'reliability_reg_weight', 0.0)) <= 0:
            return out['logits'].new_zeros(())
        alpha = out['reliability_alpha']
        mask = out['modality_mask'].to(alpha.dtype)
        target = float(getattr(self.cfg, 'reliability_target', 0.9))
        denom = mask.sum().clamp(min=1.0)
        return ((alpha - target) ** 2 * mask).sum() / denom

    def _residual_scale_value(self) -> torch.Tensor:
        if self.residual_head is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        if self.residual_scale_logit is not None:
            return torch.sigmoid(self.residual_scale_logit)
        return self.residual_scale_fixed.to(next(self.parameters()).device)

    def _fuse_poe(self, mu_mod: torch.Tensor, logvar_mod: torch.Tensor, mod_mask: torch.Tensor, reliability_alpha: Optional[torch.Tensor]=None):
        B, M, d = mu_mod.shape
        raw_mask = mod_mask.to(mu_mod.dtype)
        if reliability_alpha is None:
            reliability_alpha = torch.ones(B, M, device=mu_mod.device, dtype=mu_mod.dtype)
        eff_mask = raw_mask * reliability_alpha.to(mu_mod.dtype)
        mask = eff_mask.unsqueeze(-1)
        prec_mod = torch.exp(-logvar_mod)
        eta_mod = prec_mod * mu_mod
        prec_obs_sum = (prec_mod * mask).sum(dim=1)
        eta_obs_sum = (eta_mod * mask).sum(dim=1)
        has_any = (mod_mask.sum(dim=1, keepdim=True) > 0).to(mu_mod.dtype)
        c_hat_obs, _ = self._poe_from_nat(prec_obs_sum + 1e-06, eta_obs_sum)
        c_hat_obs = c_hat_obs * has_any
        fusion_mode = str(getattr(self.cfg, 'fusion_mode', 'poe')).lower().strip()
        if fusion_mode == 'deterministic':
            obs_count = raw_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            mu_fuse = (mu_mod * raw_mask.unsqueeze(-1)).sum(dim=1) / obs_count
            mu_fuse = mu_fuse * has_any
            logvar_fuse = torch.zeros_like(mu_fuse)
            mu_prior = torch.zeros_like(mu_fuse)
            logvar_prior = torch.zeros_like(mu_fuse)
            kl_per_mod_per_dim = torch.zeros(B, M, d, device=mu_mod.device, dtype=mu_mod.dtype)
            kl_fuse_per_dim = torch.zeros(B, d, device=mu_mod.device, dtype=mu_mod.dtype)
            return {'mu_fuse': mu_fuse, 'logvar_fuse': logvar_fuse, 'mu_prior': mu_prior, 'logvar_prior': logvar_prior, 'kl_fuse_per_dim': kl_fuse_per_dim, 'kl_per_mod_per_dim': kl_per_mod_per_dim, 'c_hat_obs': c_hat_obs, 'reliability_alpha': reliability_alpha, 'effective_modality_weight': raw_mask}
        mu_prior = self.prior.prior_mean_from_point_estimate(c_hat_obs)
        logvar_prior = self.prior.logvar_vector().unsqueeze(0).expand(B, -1)
        prec_prior = torch.exp(-logvar_prior)
        eta_prior = prec_prior * mu_prior
        prec_fuse = prec_prior + (prec_mod * mask).sum(dim=1)
        eta_fuse = eta_prior + (eta_mod * mask).sum(dim=1)
        mu_fuse, logvar_fuse = self._poe_from_nat(prec_fuse, eta_fuse)
        kl_per_mod_per_dim = torch.stack([self.kl_diag_gaussians_per_dim(mu_mod[:, m], logvar_mod[:, m], mu_prior, logvar_prior) for m in range(M)], dim=1)
        kl_fuse_per_dim = self.kl_diag_gaussians_per_dim(mu_fuse, logvar_fuse, mu_prior, logvar_prior)
        return {'mu_fuse': mu_fuse, 'logvar_fuse': logvar_fuse, 'mu_prior': mu_prior, 'logvar_prior': logvar_prior, 'kl_fuse_per_dim': kl_fuse_per_dim, 'kl_per_mod_per_dim': kl_per_mod_per_dim, 'c_hat_obs': c_hat_obs, 'reliability_alpha': reliability_alpha, 'effective_modality_weight': eff_mask}

    def forward(self, x: Optional[torch.Tensor], modality_mask: Optional[torch.Tensor]=None, return_dict: bool=False, sample: Optional[bool]=None, precomputed_encodings: Optional[Tuple[torch.Tensor, torch.Tensor]]=None):
        if precomputed_encodings is not None:
            mu_mod, logvar_mod = precomputed_encodings
            B = mu_mod.shape[0]
            device = mu_mod.device
            dtype = mu_mod.dtype
        else:
            assert x is not None, 'x is required unless precomputed_encodings is supplied'
            B = x.shape[0]
            device = x.device
            dtype = x.dtype
            mu_mod, logvar_mod = self._encode_all(x)
        if modality_mask is None:
            modality_mask = torch.ones(B, self.n_modalities, device=device, dtype=dtype)
        reliability_alpha = self._compute_reliability_alpha(mu_mod, logvar_mod)
        fused = self._fuse_poe(mu_mod, logvar_mod, modality_mask, reliability_alpha=reliability_alpha)
        mu_fuse = fused['mu_fuse']
        if sample is None:
            sample = bool(self.cfg.prior_sample and self.training)
        if sample:
            std = torch.exp(0.5 * fused['logvar_fuse'])
            z = mu_fuse + std * torch.randn_like(std)
        else:
            z = mu_fuse
        z_norm = self.concept_ln(z)
        if self.training:
            z_norm = self.concept_dropout(z_norm)
        logits_concept = self.classifier(z_norm)
        logits_residual = None
        residual_scale = logits_concept.new_zeros(())
        if self.residual_head is not None:
            if x is None:
                raise RuntimeError('Residual direct path requires x. Pass x even when using precomputed_encodings.')
            x_masked = _masked_concat_input(x, modality_mask, int(self.cfg.in_dim_per_modality))
            logits_residual = self.residual_head(x_masked)
            residual_scale = self._residual_scale_value().to(device=logits_concept.device, dtype=logits_concept.dtype)
            logits = float(getattr(self.cfg, 'concept_logit_weight', 1.0)) * logits_concept + residual_scale * logits_residual
        else:
            logits = float(getattr(self.cfg, 'concept_logit_weight', 1.0)) * logits_concept
        biology_out: Dict[str, torch.Tensor] = {}
        if self.enable_biology_heads and self.biology_heads is not None:
            assert self.biology_slice is not None
            p_slice = self.biology_slice
            p_block = z[:, p_slice]
            amy_logit, tau_logit = self.biology_heads(p_block)
            biology_out['amy_logit'] = amy_logit
            biology_out['tau_logit'] = tau_logit
        if not return_dict:
            return logits
        return {'logits': logits, 'logits_concept': logits_concept, 'logits_residual': logits_residual if logits_residual is not None else torch.zeros_like(logits_concept), 'residual_scale': residual_scale.detach(), 'concept_mean': mu_fuse, 'concept_logvar': fused['logvar_fuse'], 'prior_mean': fused['mu_prior'], 'prior_logvar': fused['logvar_prior'], 'kl_fuse_per_dim': fused['kl_fuse_per_dim'], 'kl_per_mod_per_dim': fused['kl_per_mod_per_dim'], 'mu_mod': mu_mod, 'logvar_mod': logvar_mod, 'modality_mask': modality_mask, 'reliability_alpha': fused['reliability_alpha'], 'effective_modality_weight': fused['effective_modality_weight'], 'anchor_pred': self.anchor_head(mu_fuse), **biology_out}

    def classifier_l1(self) -> torch.Tensor:
        if isinstance(self.classifier, nn.Linear):
            return self.classifier.weight.abs().sum()
        last_linear = None
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        assert last_linear is not None
        return last_linear.weight.abs().sum()

    def attach_specialist_proj(self, pairs: List[Tuple[int, int]], proj_dim: int=32) -> None:
        """Create projection heads for the specialist InfoNCE loss."""
        if not pairs:
            self.specialist_proj = None
            return
        self.specialist_proj = SpecialistProjections(group_dims=self.group_dims, pairs=pairs, proj_dim=proj_dim).to(next(self.parameters()).device)

    def dag_consistency_loss(self, mu_mod: torch.Tensor, modality_mask: torch.Tensor) -> torch.Tensor:
        """Softly align each group with the prediction from its parent groups.

        This is optional and should be used as a regularizer / ablation, not as a
        claim of identified causality.
        """
        B, M, d = mu_mod.shape
        mask = modality_mask.to(mu_mod.dtype)
        group_point = mu_mod.new_zeros(B, len(self.group_dims), d)
        group_weight = mu_mod.new_zeros(B, len(self.group_dims))
        target = self.modality_group_targets.to(mu_mod.device, dtype=mu_mod.dtype)
        for m_idx in range(M):
            w_m = mask[:, m_idx].unsqueeze(-1)
            for g_idx, sl in enumerate(self.group_slices):
                w = w_m * target[m_idx, g_idx]
                group_point[:, g_idx, sl] += w * mu_mod[:, m_idx, sl]
                group_weight[:, g_idx] += mask[:, m_idx] * target[m_idx, g_idx]
        group_weight = group_weight.clamp(min=1e-06)
        c_point = mu_mod.new_zeros(B, d)
        for g_idx, sl in enumerate(self.group_slices):
            c_point[:, sl] = group_point[:, g_idx, sl] / group_weight[:, g_idx].unsqueeze(-1)
        dag_mean = self.prior.prior_mean_from_point_estimate(c_point)
        total = mu_mod.new_zeros(())
        n_terms = 0
        for g_idx, parents in enumerate(self.group_parents):
            if len(parents) == 0:
                continue
            sl = self.group_slices[g_idx]
            pred = c_point[:, sl]
            target_group = dag_mean[:, sl].detach()
            total = total + ((pred - target_group) ** 2).mean()
            n_terms += 1
        if n_terms == 0:
            return total
        return (total / n_terms).clamp(max=self.cfg.dag_max_scale)

    def _group_precision_scores(self, out: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return group-wise precision means and sums.

        precision_mean: B x M x G, comparable across groups.
        precision_sum:  B x M x G, closer to the actual PoE natural-parameter mass.
        """
        logvar_mod = out['logvar_mod']
        precision = torch.exp(-logvar_mod)
        group_means, group_sums = ([], [])
        for sl in self.group_slices:
            p = precision[:, :, sl]
            group_means.append(p.mean(dim=-1))
            group_sums.append(p.sum(dim=-1))
        return (torch.stack(group_means, dim=-1), torch.stack(group_sums, dim=-1))

    def modality_group_alignment_loss(self, out: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Optional relaxed primary-anchor regularization.

        The forward model always allows every modality to provide evidence to
        every concept block. This loss is only an optional soft orientation.

        Supported modes:
          - kl: original precision-distribution KL over groups for each modality.
          - margin: primary group probability should exceed other groups by delta.
          - contrib_kl: align actual PoE contribution over modalities for each group.
          - contrib_margin: primary owner contribution should exceed other modalities.
          - owner_min: primary owner contribution should exceed a small absolute rho.
          - margin_contrib: margin + contrib_margin.

        All terms are missing-aware. Observed modality masks gate per-modality
        losses, and contribution-level losses are skipped for a group when the
        primary owner modality is absent.
        """
        mode = str(getattr(self.cfg, 'group_align_mode', 'kl')).lower().strip()
        if mode in {'', 'none', 'off'}:
            return out['logits'].new_zeros(())
        mode_tokens = set([t.strip() for t in mode.replace(',', '+').split('+') if t.strip()])
        if mode == 'margin_contrib':
            mode_tokens = {'margin', 'contrib_margin'}
        if mode == 'kl_margin':
            mode_tokens = {'kl', 'margin'}
        mask = out['modality_mask'].to(out['mu_mod'].dtype)
        group_mean, group_sum = self._group_precision_scores(out)
        target = self.modality_group_targets.to(group_mean.device, dtype=group_mean.dtype)
        primary = self.modality_group_primary_mask.to(group_mean.device, dtype=group_mean.dtype)
        delta = float(getattr(self.cfg, 'group_margin_delta', 0.1))
        rho = float(getattr(self.cfg, 'group_owner_min_rho', 0.35))
        floor = float(getattr(self.cfg, 'group_contrib_floor', getattr(self.cfg, 'group_prior_floor', 0.02)))
        losses: List[torch.Tensor] = []
        if 'kl' in mode_tokens:
            pred_dist = torch.softmax(torch.log(group_mean + 1e-06), dim=-1)
            target_b = target.unsqueeze(0).expand_as(pred_dist)
            kl = F.kl_div(torch.log(pred_dist + 1e-08), target_b, reduction='none').sum(dim=-1)
            denom = mask.sum().clamp(min=1.0)
            losses.append((kl * mask).sum() / denom)
        if 'margin' in mode_tokens:
            probs = group_mean / group_mean.sum(dim=-1, keepdim=True).clamp(min=1e-08)
            prim_count = primary.sum(dim=-1).clamp(min=1.0)
            prim_score = (probs * primary.unsqueeze(0)).sum(dim=-1) / prim_count.unsqueeze(0)
            nonprimary = (1.0 - primary).unsqueeze(0)
            non_scores = probs.masked_fill(nonprimary < 0.5, -1000000000.0)
            max_non = non_scores.max(dim=-1).values.clamp(min=0.0)
            margin_loss = F.relu(delta - (prim_score - max_non))
            denom = mask.sum().clamp(min=1.0)
            losses.append((margin_loss * mask).sum() / denom)
        if 'contrib_kl' in mode_tokens or 'contrib_margin' in mode_tokens or 'owner_min' in mode_tokens:
            alpha = out.get('reliability_alpha')
            if alpha is None:
                alpha = torch.ones_like(mask)
            eff = mask * alpha.to(mask.dtype)
            contrib_num = group_sum * eff.unsqueeze(-1)
            contrib_den = contrib_num.sum(dim=1, keepdim=True).clamp(min=1e-08)
            contrib = contrib_num / contrib_den
            primary_gm = primary.transpose(0, 1)
            owner_present = (eff.unsqueeze(1) * primary_gm.unsqueeze(0)).sum(dim=-1) > 0.0
            if 'contrib_kl' in mode_tokens:
                target_mg = torch.where(primary > 0.5, torch.ones_like(primary), torch.full_like(primary, floor))
                target_bmg = target_mg.unsqueeze(0) * eff.unsqueeze(-1)
                target_bmg = target_bmg / target_bmg.sum(dim=1, keepdim=True).clamp(min=1e-08)
                kl = (target_bmg * (torch.log(target_bmg + 1e-08) - torch.log(contrib + 1e-08))).sum(dim=1)
                weight = owner_present.to(group_mean.dtype)
                denom = weight.sum().clamp(min=1.0)
                losses.append((kl * weight).sum() / denom)
            if 'contrib_margin' in mode_tokens:
                primary_bmg = primary.unsqueeze(0)
                owner_contrib = (contrib * primary_bmg).sum(dim=1)
                non_owner_mask = (1.0 - primary).unsqueeze(0)
                non_contrib = contrib.masked_fill(non_owner_mask < 0.5, -1000000000.0)
                max_non = non_contrib.max(dim=1).values.clamp(min=0.0)
                margin_loss = F.relu(delta - (owner_contrib - max_non))
                weight = owner_present.to(group_mean.dtype)
                denom = weight.sum().clamp(min=1.0)
                losses.append((margin_loss * weight).sum() / denom)
            if 'owner_min' in mode_tokens:
                primary_bmg = primary.unsqueeze(0)
                owner_contrib = (contrib * primary_bmg).sum(dim=1)
                owner_min_loss = F.relu(rho - owner_contrib)
                weight = owner_present.to(group_mean.dtype)
                denom = weight.sum().clamp(min=1.0)
                losses.append((owner_min_loss * weight).sum() / denom)
        if not losses:
            return out['logits'].new_zeros(())
        return torch.stack(losses).sum()


def anchor_mse_loss(anchor_pred: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    valid = torch.isfinite(anchors)
    if valid.sum() == 0:
        return anchor_pred.new_zeros(())
    return F.mse_loss(anchor_pred[valid], anchors[valid])


def compute_ccbpoe_loss(head: CCBPoEHead, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, anchors: torch.Tensor, ce: nn.Module, *, kl_weight: float, dag_weight: float, group_prior_weight: float, anchor_weight: float, free_bits: float, l1_weight: float, n_extra_masks: int=0, extra_mask_weight: float=0.0, cascade_weight: float=0.0, cascade_modalities: Sequence[str]=(), cascade_scheme: str='rpns', cascade_group_dims: Sequence[int]=(), specialist_weight: float=0.0, specialist_temperature: float=0.2, rng: Optional[torch.Generator]=None, amy_label: Optional[torch.Tensor]=None, amy_label_mask: Optional[torch.Tensor]=None, tau_label: Optional[torch.Tensor]=None, tau_label_mask: Optional[torch.Tensor]=None, biology_lambda_amy: float=0.0, biology_lambda_tau: float=0.0, biology_biospec_downweight: float=0.5) -> Tuple[torch.Tensor, Dict[str, float]]:
    out = head(x, modality_mask=mask, return_dict=True)
    logits = out['logits']
    loss_ce = ce(logits, y)
    kl_per_dim = out['kl_fuse_per_dim'].mean(dim=0)
    loss_kl = torch.clamp(kl_per_dim, min=free_bits).sum()
    loss_dag = head.dag_consistency_loss(out['mu_mod'], mask) if dag_weight > 0 else logits.new_zeros(())
    loss_group = head.modality_group_alignment_loss(out) if group_prior_weight > 0 else logits.new_zeros(())
    loss_anchor = anchor_mse_loss(out['anchor_pred'], anchors) if anchor_weight > 0 else logits.new_zeros(())
    loss_reliability = head.reliability_regularization_loss(out)
    loss_l1 = head.classifier_l1()
    if specialist_weight > 0 and getattr(head, 'specialist_proj', None) is not None:
        loss_specialist = specialist_infonce_loss(out, head.specialist_proj, mask, temperature=specialist_temperature)
    else:
        loss_specialist = logits.new_zeros(())
    if cascade_weight > 0 and cascade_modalities:
        loss_cascade = _cascade_kl_loss_v2(out, mask, cascade_scheme, cascade_modalities, cascade_group_dims, prior_module=getattr(head, '_prior_module_runtime', None))
    else:
        loss_cascade = logits.new_zeros(())
    loss_amy = logits.new_zeros(())
    loss_tau = logits.new_zeros(())
    if getattr(head, 'enable_biology_heads', False) and 'amy_logit' in out and ('tau_logit' in out):
        biospec_input_mask = mask[:, 3].to(logits.dtype)
        biospec_w = torch.where(biospec_input_mask >= 0.5, torch.ones_like(biospec_input_mask), torch.full_like(biospec_input_mask, float(biology_biospec_downweight)))
        bio_cfg = getattr(head, '_biology_config_runtime', None)
        bio_use_pos_weight = bool(getattr(head, '_biology_use_pos_weight_runtime', False))
        bio_label_smoothing = float(getattr(head, '_biology_label_smoothing_runtime', 0.0))
        bio_focal_gamma = float(getattr(head, '_biology_focal_gamma_runtime', 0.0))
        amy_pos_weight = float(getattr(head, '_biology_amy_pos_weight_runtime', 1.0)) if bio_use_pos_weight else 1.0
        tau_pos_weight = float(getattr(head, '_biology_tau_pos_weight_runtime', 1.0)) if bio_use_pos_weight else 1.0
        if bio_cfg is not None:
            bio_label_smoothing = float(getattr(bio_cfg, 'label_smoothing', bio_label_smoothing))
            bio_focal_gamma = float(getattr(bio_cfg, 'focal_gamma', bio_focal_gamma))
        if float(biology_lambda_amy) > 0 and amy_label is not None and (amy_label_mask is not None):
            amy_w = amy_label_mask.to(logits.dtype) * biospec_w
            if amy_w.sum() > 0:
                amy_target = amy_label.clamp(min=0).to(logits.dtype)
                loss_amy = _flexible_bce_loss(out['amy_logit'], amy_target, amy_w, pos_weight=amy_pos_weight, label_smoothing=bio_label_smoothing, focal_gamma=bio_focal_gamma)
        if float(biology_lambda_tau) > 0 and tau_label is not None and (tau_label_mask is not None):
            tau_w = tau_label_mask.to(logits.dtype) * biospec_w
            if tau_w.sum() > 0:
                tau_target = tau_label.clamp(min=0).to(logits.dtype)
                loss_tau = _flexible_bce_loss(out['tau_logit'], tau_target, tau_w, pos_weight=tau_pos_weight, label_smoothing=bio_label_smoothing, focal_gamma=bio_focal_gamma)
    loss_biology_ortho = logits.new_zeros(())
    biology_ortho_weight = float(getattr(head, '_biology_orthogonality_weight_runtime', 0.0))
    if biology_ortho_weight > 0 and getattr(head, 'biology_heads', None) is not None:
        try:
            amy_w0 = head.biology_heads.amyloid_head[0].weight
            tau_w0 = head.biology_heads.tau_head[0].weight
            amy_w0 = F.normalize(amy_w0, dim=1, eps=1e-08)
            tau_w0 = F.normalize(tau_w0, dim=1, eps=1e-08)
            loss_biology_ortho = (amy_w0 @ tau_w0.T).pow(2).mean()
        except Exception:
            loss_biology_ortho = logits.new_zeros(())
    loss_extra = logits.new_zeros(())
    if n_extra_masks > 0 and extra_mask_weight > 0:
        extras: List[torch.Tensor] = []
        mu_mod, logvar_mod = (out['mu_mod'], out['logvar_mod'])
        B = x.shape[0]
        for _ in range(int(n_extra_masks)):
            keep = torch.rand(B, head.n_modalities, generator=rng, device=x.device) > 0.5
            sub_mask = mask * keep.float()
            all_zero = sub_mask.sum(dim=1) < 0.5
            if all_zero.any():
                sub_mask[all_zero] = mask[all_zero]
            out_m = head(x=x, modality_mask=sub_mask, return_dict=True, precomputed_encodings=(mu_mod, logvar_mod))
            extras.append(ce(out_m['logits'], y))
        loss_extra = torch.stack(extras).mean()
    total = loss_ce + extra_mask_weight * loss_extra + kl_weight * loss_kl + dag_weight * loss_dag + group_prior_weight * loss_group + anchor_weight * loss_anchor + float(getattr(head.cfg, 'reliability_reg_weight', 0.0)) * loss_reliability + l1_weight * loss_l1 + specialist_weight * loss_specialist + cascade_weight * loss_cascade + float(biology_lambda_amy) * loss_amy + float(biology_lambda_tau) * loss_tau + biology_ortho_weight * loss_biology_ortho
    parts = {'loss_total': float(total.detach().cpu().item()), 'loss_ce': float(loss_ce.detach().cpu().item()), 'loss_extra': float(loss_extra.detach().cpu().item()), 'loss_kl': float(loss_kl.detach().cpu().item()), 'loss_dag': float(loss_dag.detach().cpu().item()), 'loss_group': float(loss_group.detach().cpu().item()), 'loss_anchor': float(loss_anchor.detach().cpu().item()), 'loss_reliability': float(loss_reliability.detach().cpu().item()), 'loss_specialist': float(loss_specialist.detach().cpu().item()), 'loss_cascade': float(loss_cascade.detach().cpu().item()), 'loss_l1': float(loss_l1.detach().cpu().item()), 'loss_amy': float(loss_amy.detach().cpu().item()), 'loss_tau': float(loss_tau.detach().cpu().item()), 'loss_biology_ortho': float(loss_biology_ortho.detach().cpu().item())}
    return (total, parts)
