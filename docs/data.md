# Data and input formats

[← README](../README.md) · [Training](training.md) · [Reproducibility](reproducibility.md)

CoPoE consumes frozen embeddings from four modalities. Raw ADNI data, participant identifiers, patient-level arrays, and trained weights are not included. Obtain ADNI data through the study's authorized access process; keep your local data and outputs outside the public source tree.

## Split contract

Assign subjects to the official fusion **train / validation / test** splits before fitting preprocessing or encoders. The same participant must never occur across those splits, including through repeated visits.

For stage-1 model selection, divide only the official training subjects into `stage1_train` and `stage1_inner_val`. Train each modality encoder on subjects with that modality. After selecting its checkpoint, export embeddings for:

- `fusion_train`: all official training subjects with the modality, including stage-1 inner-validation subjects;
- `fusion_val`: the untouched official validation subjects with the modality;
- `fusion_test`: the untouched official test subjects with the modality.

The stage-1 trainer checks its train/inner-validation ID overlap. The embedding assembler checks cross-split overlap across all four modalities. These checks do not establish whether a supplied feature was fitted using held-out data: keep feature selection, imputation, scaling, and encoder checkpoint selection within the appropriate training split.

The manuscript's cohort has 3,018 training subjects (992 complete and 2,026 incomplete), plus 100 complete-case validation and 200 complete-case test subjects. Supplying only 992 complete-case training subjects changes this protocol. The release does not construct these subject lists or specify a visit-selection/aggregation rule; supply consistently prepared, split-assigned subject-level inputs.

## Stage-1 inputs

Each modality/split uses an `.npz` archive with these arrays:

| Array | Shape and type | Meaning |
| --- | --- | --- |
| `ids` | `[N]`, Unicode/string | Unique subject identifiers; use the same identifier across modalities. |
| `y` | `[N]`, integer | Diagnosis: **CN=0, MCI=1, AD=2**. |
| `x`, image | `[N,1,D,H,W]`, float32 | Preprocessed single-channel MRI; each spatial dimension must be at least 16. |
| `x`, other modalities | `[N,F]`, float32 | Preprocessed genomic, clinical, or biospecimen features. |
| `missing_mask`, optional | `[N,F]`, binary | Genomics only: **1 means missing** and excludes that value from VAE reconstruction loss. |

All `x` entries must be finite. Fill individual missing features using training-fitted preprocessing before calling the encoder; this is separate from withholding an entire modality during PoE fusion. The paper reports MRI inputs of `96 × 112 × 96` and 16,384 selected SNPs; the public loader also accepts other valid feature sizes for new experiments.

For large MRI inputs, pass a directory containing `ids.npy`, `x.npy`, and `y.npy` instead of an archive. An optional `missing_mask.npy` follows the same contract. The directory loader memory-maps arrays and requires `x.npy` to already be float32. It does not enable NumPy pickle loading.

The public package starts from these preprocessed arrays. MRI preprocessing, raw ADNI table joins, visit matching, genomic selection, categorical encoding, and historical data augmentations must be performed separately; see [release scope](reproducibility.md#what-is-included).

## Frozen embedding files

`copoe.encode` produces one file per modality and official split:

Place these files in `embeddings/`:

| Modality | Training | Validation | Test |
| --- | --- | --- | --- |
| Image | `image_train.npz` | `image_val.npz` | `image_test.npz` |
| Genomic | `genomic_train.npz` | `genomic_val.npz` | `genomic_test.npz` |
| Clinical | `clinical_train.npz` | `clinical_val.npz` | `clinical_test.npz` |
| Biospecimen | `biospecimen_train.npz` | `biospecimen_val.npz` | `biospecimen_test.npz` |

Each file contains `ids` and `y` as above, plus `embedding`: a finite float32 array of shape `[N,128]`. Include only subjects whose modality is available. Each subject's diagnosis must agree across modality files.

An optional PET CSV supplies supervision labels with columns `id,amy,tau`. Observed labels are `0` or `1`; leave a value blank when unavailable. PET labels are auxiliary targets, never fusion input features. A synthetic format example is:

```csv
id,amy,tau
synthetic_0001,1,0
synthetic_0002,,1
synthetic_0003,0,
```

Prepare fusion inputs with:

```bash
python -m copoe.prepare \
  --embeddings-dir embeddings \
  --pet-labels private/pet_labels.csv \
  --out data
```

Omit `--pet-labels` when preparing a diagnosis-only dataset. The output directory must be new or empty. The assembler:

1. Rejects cross-split participant overlap, duplicate IDs, and conflicting labels for assembled subjects.
2. Uses the **union** of subjects for training and the **intersection** of all four modalities for validation/test.
3. Fits each embedding dimension's mean and population standard deviation on observed training subjects only; constant dimensions use a standard deviation of 1.
4. Standardizes observed embeddings and zero-fills absent modality blocks.
5. Writes `train.npz`, `val_all4.npz`, `test_all4.npz`, `normalizer.npz`, and `manifest.json`.

`manifest.json` contains modality order, counts, and PET-label availability, without participant identifiers. Inspect its split counts before training. Fused output files omit IDs but still contain patient-level representations and labels; they should remain private.

For older **trusted local** archives whose IDs use NumPy object arrays, `--trusted-object-ids` enables pickle-based loading. Use ordinary Unicode arrays for newly exported files; do not enable this flag for untrusted downloads.

## Fusion input contract

If you already have correctly normalized fusion arrays, supply them directly to `copoe.train` and skip the assembler. You are then responsible for participant-disjoint splits because the fused files have no ID field to check.

| Array | Shape | Meaning |
| --- | --- | --- |
| `x` | `[N,512]`, float32 | Concatenated frozen embeddings in **I, G, C, B** order. |
| `mask` | `[N,4]`, binary | Availability in the same order; **1 means observed**. Every row needs at least one observed modality. |
| `y` | `[N]`, integer | **CN=0, MCI=1, AD=2**. |
| `amy_label`, `tau_label` | `[N]`, integer | PET positivity, `0` or `1`; use `-1` when unavailable. |
| `amy_mask`, `tau_mask` | `[N]`, binary | PET-label availability; **1 means observed**. |

The four blocks are:

| Modality | Array columns | Mask column |
| --- | --- | --- |
| Image (`I`) | `x[:, 0:128]` | `mask[:, 0]` |
| Genomic (`G`) | `x[:, 128:256]` | `mask[:, 1]` |
| Clinical (`C`) | `x[:, 256:384]` | `mask[:, 2]` |
| Biospecimen (`B`) | `x[:, 384:512]` | `mask[:, 3]` |

The paper's table display order **G, B, I, C** is different from this fixed array order. Do not reorder the input arrays to match the table.

The four PET arrays may be omitted together for a diagnosis-only run; the loader supplies unavailable labels/masks. Otherwise provide labels and masks consistently. Missing embedding blocks should be zero, and the loader also zeros them defensively. Do not include PET targets, diagnosis-defining fields, or participant IDs in `x`.

Training permits naturally incomplete profiles. Validation and the controlled 15-subset test require complete-case subjects, so every reported subset is evaluated on the same people. The evaluator withholds modalities at inference; absent modalities contribute no PoE factor.
