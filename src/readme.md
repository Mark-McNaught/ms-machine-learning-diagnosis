# README

This project investigates attention-augmented deep learning for binary classification on medical imaging data (Multiple Sclerosis classification). It systematically compares eight ResNet18-based architectures augmented with CBAM and SE attention modules against a ViT-based model (EfficientFormer-L1), and additionally evaluates two alternative classification strategies: an NCA + kNN metric-learning pipeline and a CNN-MHSA hybrid head. The work is structured as four sub-research questions (SRQs), each corresponding to a Jupyter notebook.

## File Structure

```
project_root/
│
├── data/
│   ├── raw/                          # Dataset (images, organised by class)
│   └── experiments/
│       ├── grid-search-results/      # Hyperparameter search outputs
│       │   ├── grid_search_results.csv
│       │   ├── grid_search_results_v2.csv
│       │   ├── grid_search_summary_combined.csv
│       │   └── optimal_config.csv    # ← consumed by downstream notebooks
│       ├── head-ablation-results/
│       │   ├── head_ablation_results.csv
│       │   ├── head_ablation_summary.csv
│       │   └── optimal_head.csv      # ← consumed by downstream notebooks
│       ├── arch-eval-results/
│       │   ├── arch_eval_results.csv
│       │   ├── arch_eval_summary.csv
│       │   ├── arch_eval_test_results.csv
│       │   ├── weights/<arch>/fold_<n>.pt
│       │   ├── training-curves/
│       │   └── plots/
│       ├── nca-knn-results/
│       │   ├── nca_knn_results.csv
│       │   ├── nca_knn_test_results.csv
│       │   ├── param_search_results.csv
│       │   └── plots/
│       ├── cnn-mhsa-hybrid-results/
│       │   ├── hybrid_results.csv
│       │   ├── hybrid_test_results.csv
│       │   ├── weights/
│       │   └── training-curves/
│       └── vit-comparison-results/
│           ├── vit_cv_results.csv
│           ├── vit_test_results.csv
│           ├── weights/efficientformer/fold_<n>.pt
│           ├── training-curves/
│           └── plots/
│
├── src/
│   └── scripts/
│       ├── data.py        # Dataset loading, transforms, splits, DataLoaders
│       ├── models.py      # Model definitions (ResNet18 variants, EfficientFormer, CNN-MHSA)
│       ├── trainer.py     # Training loops (two-phase protocol), feature extraction
│       ├── evaluator.py   # Test set evaluation (F1, AUC, ECE, confusion matrix)
│       └── utils.py       # Seed setting, weight I/O, run-resumption helpers
│
└── notebooks/
    ├── grid-search.ipynb         # Hyperparameter search (run first)
    ├── head-ablation.ipynb       # MLP vs linear probe ablation (run second)
    ├── arch-eval.ipynb           # SRQ1 — 8-architecture evaluation
    ├── nca-knn-eval.ipynb        # SRQ2 — NCA + kNN metric learning pipeline
    ├── cnn-mhsa-hybrid.ipynb     # SRQ3 — CNN + tokenised self-attention head
    └── vit-comparison-final.ipynb # SRQ4 — Grand comparison with EfficientFormer
```

All notebooks must be run from their own directory (two levels below the project root) so that relative `Path().resolve().parents[1]` resolves correctly to the project root.

---

## Build Instructions

### Requirements

- Python 3.10 or later
- CUDA-capable GPU strongly recommended (all notebooks support CPU fallback but training times are prohibitive without a GPU)
- Packages: listed in `requirements.txt`
- Tested on Windows 11 with only CPU-training and Python 3.11

Key dependencies include:

- `torch` / `torchvision` (PyTorch ≥ 2.0)
- `timm` (for EfficientFormer-L1)
- `scikit-learn` (cross-validation, NCA, kNN, metrics)
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `jupyter` or `jupyterlab`

### Build Steps

1. **Clone the repository and navigate to the project root.**

2. **Create and activate a virtual environment.**
   (Follow instructions in `config/readme.md`)

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run experiments.**
   The dataset will automatically be fetched on the first run — the initial experiment run will take longer due to the dataset download.

### Test Steps

Run the experiments in the order described in the user manual, monitor notebook output for per-fold progress logs, and verify results by inspecting the CSV files written to the relevant subdirectory under `data/experiments/`.