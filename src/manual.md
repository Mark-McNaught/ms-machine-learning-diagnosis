# User Manual

## Overview

The experiment pipeline is split across six Jupyter notebooks. They have a strict dependency order — each notebook reads output files produced by its predecessors. Always run them in the sequence shown below. Each notebook is independently resumable: completed folds are detected automatically via their CSV output, so a crashed or interrupted run can simply be re-run from the top of the training cell without repeating prior work.

---

## Execution Order

```
1. grid-search.ipynb          →  produces optimal_config.csv
2. head-ablation.ipynb        →  produces optimal_head.csv
3. arch-eval.ipynb            →  SRQ1, produces arch_eval_results.csv + fold weights
4. nca-knn-eval.ipynb         →  SRQ2, reads arch-eval weights
5. cnn-mhsa-hybrid.ipynb      →  SRQ3, reads arch-eval weights
6. vit-comparison-final.ipynb →  SRQ4, reads outputs from all prior stages
```

---

## Notebook-by-Notebook Guide

### 1. `grid-search.ipynb` — Hyperparameter Search

**Purpose:** Identifies the best Phase 2 fine-tuning learning rate and weight decay for the baseline ResNet18 using 3-fold cross-validation. Two successive grid searches (V1 low-LR pilot, V2 higher-LR follow-up) are combined to select a single optimal configuration.

**How to run:**
1. Open the notebook and run all cells in order, from Section 1 through Section 6.
2. Sections 4a and 4b contain the two grid search loops. Each is independently resumable. **Expected duration:** approximately 18 hours total across both searches on GPU.
3. After Section 5 completes, `optimal_config.csv` is written to `data/experiments/grid-search-results/`. This file is required by all subsequent notebooks.

**Key outputs:**
- `grid_search_results.csv` and `grid_search_results_v2.csv` — raw per-fold results
- `grid_search_summary_combined.csv` — aggregated mean ± std per configuration
- `optimal_config.csv` — single-row file with the winning `lr_phase2` and `weight_decay`

---

### 2. `head-ablation.ipynb` — MLP vs Linear Probe

**Purpose:** Determines whether the MLP projection head (`Linear(512→128) → ReLU → Dropout(0.3) → Linear(128→1)`) outperforms a minimal linear probe (`Linear(512→1)`) on the baseline ResNet18. The winning head type is used in all subsequent experiments.

**Prerequisite:** `optimal_config.csv` must exist (run grid search first).

**How to run:**
1. Run all cells in order, Sections 1 through 6.
2. The ablation loop in Section 4 runs 2 head types × 5 folds = 10 runs. **Expected duration:** 3–4 hours on GPU.
3. After Section 5 completes, `optimal_head.csv` is written to `data/experiments/head-ablation-results/`.

**Key outputs:**
- `head_ablation_results.csv` — per-fold results for each head type
- `head_ablation_summary.csv` — aggregated comparison table
- `optimal_head.csv` — single-row file identifying the winning head (`mlp` or `linear`)

---

### 3. `arch-eval.ipynb` — SRQ1: Architecture Evaluation

**Purpose:** Trains all 8 attention-augmented ResNet18 architectures under 5-fold stratified cross-validation using a two-phase transfer learning protocol. Fold weights are saved for downstream use by SRQ2 and SRQ3.

**Prerequisites:** `optimal_config.csv` and `optimal_head.csv` must exist.

**Architectures evaluated:**

| Key | Description |
|-----|-------------|
| `base` | Plain ResNet18, no attention |
| `cbam_end` | Single CBAM after avgpool |
| `cbam_block_pre` | CBAM inside each block, pre-shortcut |
| `cbam_block_post` | CBAM inside each block, post-shortcut |
| `se_end` | Single SE block after avgpool |
| `se_block_pre` | SE inside each block, pre-shortcut |
| `cbam_isolated_end` | CBAM (no shared weights) after avgpool |
| `cbam_isolated_block_pre` | CBAM (no shared weights) inside each block |

**How to run:**
1. Run all cells in order. The training loop in Section 5 runs 8 architectures × 5 folds = 40 runs. **Expected duration:** several hours to a full day on GPU depending on hardware.
2. Weights are saved automatically per fold under `data/experiments/arch-eval-results/weights/<arch>/fold_<n>.pt`.
3. Run Section 6 (Results Analysis) and Section 9 (Final Test Set Evaluation) to produce the summary and test result files.

**Key outputs:**
- `arch_eval_results.csv` — per-fold cross-validation results
- `arch_eval_summary.csv` — mean ± std per architecture
- `arch_eval_test_results.csv` — held-out test set evaluation
- `weights/<arch>/fold_<n>.pt` — saved model weights (needed by SRQ2 and SRQ3)

---

### 4. `nca-knn-eval.ipynb` — SRQ2: Metric Learning Pipeline

**Purpose:** Replaces the trained linear classifier head with a post-hoc NCA + kNN pipeline applied to 512-dimensional backbone embeddings. Evaluates whether metric-space classification improves over the linear baseline and whether the improvement varies by backbone.

**Prerequisites:** `arch_eval_results.csv` and fold weights from `arch-eval.ipynb`; `optimal_head.csv`.

**How to run:**
1. Run Sections 1–4 (setup, data, config, helpers).
2. Run Section 5 to perform parameter search (grid search over `target_dim` × `k`) on the `base` backbone. The selected parameters are then applied uniformly to all backbones.
3. Run Section 6 to evaluate all 8 backbones × 5 folds = 40 runs with the chosen parameters.
4. Run Sections 7 and 8 for CV analysis and final test set evaluation.

**Key outputs:**
- `param_search_results.csv` — NCA/kNN parameter grid results
- `nca_knn_results.csv` — per-fold CV results across backbones
- `nca_knn_test_results.csv` — held-out test set evaluation

---

### 5. `cnn-mhsa-hybrid.ipynb` — SRQ3: CNN + Tokenised Self-Attention

**Purpose:** Evaluates whether replacing the linear classifier head with a single Multi-Head Self-Attention (MHSA) layer over ViT-style spatial tokens improves MS classification. The ResNet18 backbone is frozen; only the MHSA components are trained.

**Prerequisites:** `arch_eval_results.csv` and `base` backbone fold weights from `arch-eval.ipynb`; `optimal_config.csv`; `optimal_head.csv`.

**How to run:**
1. Run all cells in order. The training loop in Section 4 runs 5 folds. Only MHSA parameters are trained (backbone frozen), so a single training phase suffices.
2. Run Section 5 for CV summary and Section 6 for final test evaluation.

**Key outputs:**
- `hybrid_results.csv` — per-fold CV results
- `hybrid_test_results.csv` — held-out test set evaluation

---

### 6. `vit-comparison-final.ipynb` — SRQ4: Grand Comparison

**Purpose:** Trains EfficientFormer-L1 under the same two-phase transfer learning protocol as the CNN experiments, then produces a final five-model comparison on the held-out test set. This is the definitive synthesis of the entire project.

**Prerequisites:** All upstream notebooks must be fully complete. Specifically:
- `optimal_config.csv` and `optimal_head.csv`
- `arch_eval_test_results.csv` (from `arch-eval.ipynb` Section 9)
- `nca_knn_test_results.csv` (from `nca-knn-eval.ipynb`)
- `hybrid_test_results.csv` (from `cnn-mhsa-hybrid.ipynb`)

**How to run:**
1. Run Sections 1–4 to load data, config, and train EfficientFormer under 5-fold CV. **Expected duration:** several hours on GPU. Gradient clipping is applied in Phase 2 (standard ViT fine-tuning practice).
2. Run Section 5 for CV summary. The best-validation-F1 fold is selected automatically.
3. Run Section 6 to evaluate all five models on the held-out test set and produce the grand comparison plots.

**Key outputs:**
- `vit_cv_results.csv` — EfficientFormer per-fold CV results
- `vit_test_results.csv` — EfficientFormer test set evaluation
- `plots/` — final comparison figures (F1, AUC-ROC, ECE across all five models)

---

## Source Module Reference (`src/scripts/`)

| Module | Purpose |
|--------|---------|
| `data.py` | `get_dataset()`, `get_classes()`, `get_paths_and_labels()`, `get_transforms()`, `get_trainval_test_split()`, `get_test_loader()` |
| `models.py` | `get_model(architecture, head)` — factory function for all supported architectures |
| `trainer.py` | Two-phase training loop, early stopping, `get_features()` for embedding extraction |
| `evaluator.py` | `evaluate_model()` — returns accuracy, precision, recall, F1, AUC-ROC, ECE, confusion matrix, and classification report |
| `utils.py` | `set_seed()`, `load_completed_runs()` (resume support), `weights_path_for()`, `load_weights()` |

---

## Important Notes

- **Outer split reproducibility:** All notebooks apply a fixed seed-42 stratified 80/20 train+val / test split. The held-out test set is identical across all experiments and is never touched until the final evaluation cell of each notebook.
- **Resuming interrupted runs:** Every training loop checks a CSV for already-completed `(architecture/backbone/head, fold)` pairs before starting. Simply re-run the training cell to pick up from where it stopped.
- **GPU memory:** EfficientFormer-L1 is significantly larger than ResNet18. If you encounter OOM errors in `vit-comparison-final.ipynb`, reduce `BATCH_SIZE` in Section 3.
- **Notebook working directory:** Each notebook resolves the project root as `Path().resolve().parents[1]`. Run notebooks from their location inside the `notebooks/` subdirectory, not from the project root, or path resolution will fail.