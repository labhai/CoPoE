# Publishing this repository

[← README](../README.md)

## Put the project at the repository root

Extract the release archive and upload the **contents of the `CoPoE/` directory** to your GitHub repository. `README.md`, `pyproject.toml`, `copoe/`, `docs/`, and `assets/` should be at the repository root. Do not upload only the ZIP, and do not place the entire project inside an extra `CoPoE/` subdirectory.

Include hidden source files/directories such as **`.gitignore` and `.github/`** when present. File browsers may hide them. Keep the relative paths unchanged so the README image and documentation links work.

For an existing local clone, copy the release contents into that clone while preserving its `.git/` directory. Review the files before committing:

```bash
git status --short
git add --dry-run .
```

Check that the proposed files contain only source, documentation, tests, and public assets. Exclude raw data, participant identifiers, embeddings, checkpoints, local environments, and run outputs. The release `.gitignore` covers common generated/data formats, but it cannot inspect file contents.

Once the proposed file list is correct, use your usual GitHub workflow. A normal commit/push from an existing clone is:

```bash
git add .
git diff --cached --stat
git commit -m "Prepare CoPoE public code release"
git push
```

Review the staged diff before the commit. No history rewrite or force push is needed. The GitHub web upload interface is also suitable if you include the complete directory contents and hidden source files.

## Suggested repository details

**Description**

```text
CoPoE: disease-coordinate product-of-experts fusion for multimodal learning with missing modalities.
```

**Topics**

```text
copoe product-of-experts multimodal-learning missing-modalities alzheimers-disease pytorch
```

Use a verified paper/project URL for the optional website field once available. Add the actual paper link to the README at that point; do not substitute a provisional arXiv identifier or claim that the code release has reproduced results that have not been rerun.

## Check the published landing page

Confirm that the README overview image loads, documentation links open, code blocks display normally, and the repository's test workflow completes. Run the synthetic smoke command from the README in a clean environment if you change dependencies or packaging before publication.

Before describing the package as an exact reproduction, reconcile the checkpoint-selection and KL differences recorded in [reproducibility.md](reproducibility.md#manuscript-and-implementation-details). Those are scientific protocol decisions, independent of the GitHub page layout.
