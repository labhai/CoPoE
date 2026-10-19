# Training and evaluation

[← README](../README.md) · [Data formats](data.md) · [Reproducibility](reproducibility.md)

Install the package from the repository root with `python -m pip install -e .`. The examples below assume you have prepared the inputs described in [data.md](data.md). CUDA commands require a compatible GPU/PyTorch installation; replace `--device cuda` with `--device cpu` for CPU execution.

## 1. Train and export modality encoders

Each stage-1 encoder is trained independently. Both `stage1_train` and `stage1_inner_val` must come only from official fusion training subjects. The export loop then embeds all available subjects in each official split, including the stage-1 inner-validation subjects in `fusion_train`.

```bash
for modality in image genomic clinical biospecimen; do
  python -m copoe.pretrain \
    --modality "$modality" \
    --train "prepared/${modality}_stage1_train.npz" \
    --val "prepared/${modality}_stage1_inner_val.npz" \
    --out "runs/encoders/${modality}" \
    --device cuda

  for split in train val test; do
    python -m copoe.encode \
      --modality "$modality" \
      --checkpoint "runs/encoders/${modality}/best_encoder.pt" \
      --input "prepared/${modality}_fusion_${split}.npz" \
      --output "embeddings/${modality}_${split}.npz" \
      --device cuda
  done
done
```

For memory-mapped MRI arrays, replace each image `.npz` path with the corresponding directory of `.npy` files. Stage-1 training requires a new or empty output directory, and embedding export refuses to overwrite an existing output file.

### Architectures and defaults

All four encoders export **128 dimensions**. The image encoder exports its pooled projection, clinical/biospecimen encoders their penultimate representation, and the genomic VAE its deterministic posterior mean.

| Modality | Architecture | Epochs | Batch | Learning rate |
| --- | --- | ---: | ---: | ---: |
| Image | 3D residual CNN, channel widths 24/48/96/192, two blocks per stage, global average pooling | 30 | 4 | `2e-4` |
| Genomic | VAE with 1024/512 encoder widths and mirrored decoder | 40 | 16 | `2e-4` |
| Clinical | Residual MLP, width 256, 3 blocks | 60 | 64 | `3e-4` |
| Biospecimen | Residual MLP, width 192, 2 blocks | 60 | 64 | `1e-4` |

Shared defaults: AdamW, weight decay `1e-3`, dropout `0.1`, label smoothing `0.05`, cosine schedule with 2 warmup epochs, patience 15, seed 42, and validation macro one-vs-rest AUROC checkpoint selection. `--balanced-sampler` and `--class-weighted-loss` are optional and **off by default**. The genomic training objective additionally uses observed-entry reconstruction weight `0.2` and KL weight `0.02`; validation loss contains classification CE only.

Useful stage-1 flags:

| Purpose | Flags |
| --- | --- |
| Optimization | `--epochs`, `--batch-size`, `--lr`, `--weight-decay`, `--patience`, `--seed`, `--grad-clip` |
| Checkpoint selection | `--monitor auroc`, `--monitor bacc`, or `--monitor val_loss` |
| Schedule and imbalance | `--scheduler cosine` / `none`, `--warmup-epochs`, `--balanced-sampler`, `--class-weighted-loss`, `--label-smoothing` |
| Architecture | `--base-channels` for MRI; `--hidden-dim` / `--depth` for tabular data; `--genomic-hidden-dims` for genomics; `--dropout` |
| Genomic losses | `--recon-weight`, `--kl-beta` |

`--embed-dim` must remain 128 for this fusion interface. Default AUROC selection requires all three diagnosis classes in the inner-validation split. `best_encoder.pt` stores architecture arguments, model weights, and best epoch; it does not store subject IDs. `history.csv` and `config.json` record the run's metrics/settings.

Historical raw-data processing and augmentations are not implemented in this CLI. A newly trained encoder run is therefore not, by itself, a reproduction of the manuscript's original encoders. See [reproducibility.md](reproducibility.md).

## 2. Assemble frozen embeddings

```bash
python -m copoe.prepare \
  --embeddings-dir embeddings \
  --pet-labels private/pet_labels.csv \
  --out data
```

Inspect `data/manifest.json`. Fusion uses **I, G, C, B** array order and normalization fitted on observed training embeddings only. Full input and split requirements are in [data.md](data.md).

## 3. Train fusion

```bash
python -m copoe.train \
  --train data/train.npz \
  --val data/val_all4.npz \
  --test data/test_all4.npz \
  --out runs/main \
  --seeds 42 43 44 45 46 \
  --device cuda
```

Fusion defaults are 50 epochs, batch size 64, and patience 16. `--test` is optional and is evaluated only after a validation-selected checkpoint is saved. Each seed writes `best.pt`, `history.csv`, and, when test data are supplied, `test_subsets.csv`. `summary.json` collects seed-level summaries. Use a fresh output directory for each experiment.

**Checkpoint selection:** the default `--selection paper` uses **all-four validation BAcc + 0.5 × amyloid AUROC + 0.5 × tau AUROC**, all on the 0–1 scale. The argument name is retained from the supplied code; it does **not** implement the manuscript's stated mean BAcc over 15 validation subsets. `--selection bacc` selects by all-four validation BAcc alone. Neither mode selects by mean 15-subset BAcc. See the [known protocol differences](reproducibility.md#manuscript-and-implementation-details).

The scoring helper substitutes an AUROC of `0.5` when a PET AUROC is undefined. The CLI's label-support checks below still apply; this fallback does not permit a full PET-enabled run without PET labels.

With PET heads enabled, the CLI requires positive and negative labels for **each** PET target in both training and validation, even when using `--selection bacc`. If PET labels are unavailable, run an explicitly diagnosis-only variant:

```bash
python -m copoe.train \
  --train data/train.npz \
  --val data/val_all4.npz \
  --test data/test_all4.npz \
  --out runs/diagnosis_only \
  --selection bacc \
  --diagnosis-only
```

This disables PET heads and their losses; it is not the full CoPoE configuration reported in the paper. The random-subset, alignment, KL, cascade, and residual components remain active.

## 4. Evaluate all 15 subsets

```bash
python -m copoe.evaluate \
  --checkpoint runs/main/seed42/best.pt \
  --data data/test_all4.npz \
  --output runs/main/seed42/test_subsets.csv \
  --device cuda
```

The input must contain complete-case subjects. Rows are ordered singles (`G`, `B`, `I`, `C`), pairs, triples, and all four (`G+B+I+C`). The same subjects are scored in every row.

CSV metrics include BAcc, macro-F1, macro one-vs-rest AUROC, 15-bin ECE, multiclass Brier score, and NLL. **BAcc, F1, AUROC, and AUPRC are fractions in the CSV; multiply by 100 when comparing with percentage columns in the paper.** Brier sums squared class-probability errors before averaging over subjects. PET AUPRC is computed as average precision. An undefined AUROC is represented as `NaN`.

When a checkpoint has PET heads, `amy_*` / `tau_*` columns evaluate those built-in heads. They do not reproduce the independently trained block-specific probes in Table III. The evaluator does not calculate bootstrap confidence intervals, significance stars, or baseline comparisons.

## Checkpoint loading

New release checkpoints use PyTorch's `weights_only=True` loading path. For trusted historical local files only:

- `copoe.encode --trusted-legacy` permits the original encoder checkpoint format.
- `copoe.evaluate --legacy-checkpoint` permits the original fusion checkpoint format.

These flags enable pickle-based loading, which can execute code. Do not use them for untrusted checkpoints. Original checkpoints may also contain participant identifiers; do not publish them directly. Architecture compatibility and scientific reproduction should be checked against the particular original checkpoints you hold.

## CLI help

```bash
python -m copoe.pretrain --help
python -m copoe.encode --help
python -m copoe.prepare --help
python -m copoe.train --help
python -m copoe.evaluate --help
python -m copoe.demo --help
```
