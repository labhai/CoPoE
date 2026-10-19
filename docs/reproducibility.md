# Reproducibility and release scope

[← README](../README.md) · [Data formats](data.md) · [Training](training.md)

This release provides an executable implementation and a synthetic smoke workflow. It does not distribute the ADNI inputs, frozen patient-level embeddings, participant split lists, or original trained checkpoints needed to independently reproduce the paper's numbers. The release review did not have those original inputs/checkpoints and therefore did not re-verify the reported ADNI performance.

## What is included

| Component | Release coverage |
| --- | --- |
| Stage-1 encoders | MRI residual CNN, clinical/biospecimen residual MLPs, genomic VAE; training and frozen embedding export CLIs |
| Fusion | Full-RPNS modality experts, masked Gaussian PoE, classification residual, PET auxiliary heads, training losses |
| Split assembly | ID joins, participant-overlap checks, training-only embedding normalization, complete-case validation/test |
| Evaluation | Diagnosis/probability-quality metrics and built-in PET head metrics over all 15 non-empty subsets |
| Software checks | Synthetic inputs and tests; these are not ADNI benchmark results |

The release does **not** include raw-data preprocessing, historical data augmentation, the eight comparison baselines, original weights, the separate block-probing pipeline, the full ablation runner, patient-level bootstrap tests, or Wilcoxon/Holm significance analysis.

The supplied repository description identifies additional historical stage-1 processing: clinical/biospecimen missingness indicators and date-column filtering, genomic SNP selection/SMOTE/chip-aware latent mixup, and image augmentation. These are not recreated by the public stage-1 CLI. Starting with differently prepared arrays or retraining with this streamlined CLI constitutes a new experiment.

## Manuscript and implementation details

Two differences require reconciliation before calling a new run an exact reproduction of the manuscript protocol. They are documented here rather than silently changing the supplied training objective or selection rule.

| Item | Manuscript description | Current implementation |
| --- | --- | --- |
| Fusion checkpoint selection | Mean validation BAcc across 15 modality subsets | Default `--selection paper`: all-four validation BAcc + `0.5 × amyloid AUROC + 0.5 × tau AUROC`. `--selection bacc`: all-four BAcc only. |
| Gaussian KL regularization | Per-modality and fused posteriors regularized toward the prior | The active objective uses **fused-posterior KL only**, averaged over the minibatch, then clamped per dimension by free bits and summed. Per-modality KL values are computed but not added to this loss. |

Code locations: [`validation_score`](../copoe/train.py) and [`compute_ccbpoe_loss`](../copoe/model.py). The original README described the first criterion as the saved main-run setting, but the original checkpoints were not available during this release review to establish their provenance.

Additional implementation details matter for reproduction:

- The classifier includes an L1 penalty (`1e-5`) in addition to the seven losses summarized in the manuscript's Eq. (7).
- PET supervision weights examples without biospecimen inputs by `0.5` relative to those with biospecimen inputs before normalization within the supervised batch.
- The cascade regularizer normalizes **mean precision per block**, not the sum over dimensions. Its fixed target vectors are `[0.40,0.40,0.10,0.10]` for G and `[0.05,0.20,0.30,0.45]` for C.
- The training sampler combines inverse diagnosis frequency with a factor of 2 for complete-case subjects. Classification CE class weights are computed from an initial sampler pass, rather than directly from unresampled cohort counts.
- The paper displays modalities as G/B/I/C; the code input contract is **I/G/C/B**. This is a representation-order convention, not a change to the modality set.

## Default fusion configuration

The effective configuration is defined by [`config.py`](../copoe/config.py), [`train.py`](../copoe/train.py), and [`model.py`](../copoe/model.py).

| Setting | Value |
| --- | --- |
| Seeds | `42, 43, 44, 45, 46` |
| Input | Four frozen 128-d embeddings, concatenated as I/G/C/B |
| Latent partition | R/P/N/S = `8/16/16/8` (48 dimensions) |
| Gaussian expert MLP hidden width / dropout | `64` / `0.15` |
| Fused representation dropout | `0.2` |
| RPNS classifier hidden width | `32` |
| Residual classifier hidden width / dropout / scale | `128` / `0.15` / `0.5` |
| Prior | Zero-mean diagonal Gaussian with learned log variances; no cascade graph edges |
| PET heads | P block only; hidden width `64`, depth `3`, dropout `0.2` |
| Optimizer | AdamW; base learning rate `7.5e-5`; weight decay `5e-4` |
| PET head learning rates | Amyloid `1.5e-4`; tau `3e-4` |
| Schedule | 5-epoch linear warmup from `0.2 × lr`, then cosine decay |
| Batch / epochs / patience | `64` / `50` / `16` |
| Earliest checkpoint | Epoch `min(10, requested epochs)` |
| Gradient clipping | Maximum norm `1.0` |
| Diagnosis label smoothing | `0.05` |
| PET focal gamma / label smoothing | `1.0` / `0.05` |
| PET positive-class weights | Negative / positive counts among observed training labels |
| Amyloid / tau loss weights | `0.3` / `0.7` |
| Fused KL weight / free bits | Up to `5e-4` / `0.02` per dimension |
| Block-alignment weight / temperature | Up to `1.0` / `0.2` |
| Cascade weight | Up to `0.1`, on C and G |
| Random-subset CE weight | `0.25`, one additional subset per example |
| Classifier L1 weight | `1e-5` |

At epoch `e`, alignment weight is `min(1,e/5)`, cascade weight is `0.1 × min(1,e/5)`, and KL weight is `5e-4 × (0.5 + 0.5 × min(1,e/5))`. For the additional subset, each available modality is retained independently with probability `0.5`; if all are dropped, the original observed set is restored. The default does not use learned reliability weighting, a causal graph loss, or external R/N/S anchor labels. The I→N and B→P block-alignment projections remain active.

## Reported results and what they mean

The manuscript reports the following all-modality results, as mean ± standard deviation across five seeds:

| BAcc (%) | Macro-F1 (%) | AUROC (%) | ECE | Brier | NLL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 71.87 ± 0.89 | 71.92 ± 0.62 | 85.80 ± 0.33 | 0.086 ± 0.011 | 0.408 ± 0.003 | 0.681 ± 0.006 |

The reported mean AUROC across the 15 subsets is **77.82%**. These values are transcribed from the paper; they are not newly reproduced results from this release review. A synthetic smoke run cannot validate them.

The baseline comparisons use standardized fusion adaptations sharing frozen embeddings, splits, and modality subsets. They should not be interpreted as benchmarks of every baseline's native end-to-end pipeline.

Table III's block-specific PET results use separately trained matched readout heads. The auxiliary PET probabilities emitted by `copoe.evaluate` are a different evaluation and should not be compared directly as a reproduction of that table. The P block is PET-supervised; this is supervised enrichment analysis, not independent validation of all four biological coordinates.

## Before reporting a reproduction

Record the data release, subject/visit selection, preprocessing, feature definitions, encoder checkpoints, fusion settings, and actual checkpoint selection criterion. Keep official held-out subjects out of preprocessing and encoder fitting. Check the assembled counts and PET label support before training, and distinguish reproduced patient-level results from software-only synthetic checks.

## Release validation

The release review passed all **15 automated tests** on Python 3.12 with PyTorch 2.5.1 (CPU), including checkpoint round-trips, analytical Gaussian PoE checks, input validation, and stage-1 training/export. Editable installation and wheel packaging succeeded. The two-epoch synthetic demo completed training and produced evaluation rows for all 15 modality subsets. CUDA execution and the remote GitHub Actions workflow have not been run during this review.
