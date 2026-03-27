# Attention-Augmented CNN Architectures for Multiple Sclerosis MRI Classification: A Comparative Study of Attention Mechanisms, Metric Learning, and Vision Transformers

**Author:** Mark McNaught (2764158M)  
**Supervisor:** Dr Chris McCaig  
**Institution:** University of Glasgow, School of Computing Science  
**Submitted:** March 2026

## Overview

Multiple sclerosis affects over 2.9 million people globally and carries a mean diagnostic delay of over 17 months, yet diagnosis remains dependent on expert MRI interpretation subject to misdiagnosis rates of 15%. This project investigates whether targeted architectural augmentations can improve automated binary MS classification from 2D axial FLAIR MRI slices over a pretrained ResNet18 baseline, under a two-phase transfer learning protocol with only 1,652 labelled slices and CPU-only compute — conditions representative of small clinical research environments.

## Aims and Research Questions

Rather than optimising a single architecture, this work systematically evaluates four design axes through four subsidiary research questions (SRQs):

- **SRQ1 — Attention Mechanisms:** Do SE-Net channel attention and CBAM each improve MS classification over a plain CNN baseline, and can the independent contributions of attention mechanism type and placement be isolated through controlled comparisons?
- **SRQ2 — Metric Learning:** Does replacing the classification head with a post-hoc NCA and k-nearest neighbours pipeline improve classification performance, and does any gain vary with backbone attention quality?
- **SRQ3 — Tokenised Self-Attention:** Does replacing the classification head with a single Multi-Head Self-Attention (MHSA) layer over ViT-style spatial tokens improve MS classification over the linear head baseline under data-scarce conditions?
- **SRQ4 — Grand Comparison:** Across the progression from plain CNN through attention augmentation, metric learning, and tokenised self-attention to a standalone Vision Transformer, where does each intervention add value, and is the complexity of a parameter-matched ViT justified under data-scarce and compute-limited conditions?

## Key Results

The plain ResNet18 baseline proved a strong foundation (F1=0.9147, AUC-ROC=0.9855). The best-performing model overall was a novel post-shortcut CBAM placement variant (`cbam_block_post`), achieving F1=0.9407, AUC-ROC=0.9883, and ECE=0.0369 — the lightest structural addition beyond the plain baseline. Performance did not increase monotonically with architectural complexity.


## Repository Structure
```
.
├── config/
│   ├── requirements.txt          # Python dependencies and version pins
│   └── README.md                 # Environment setup instructions
│
├── data/
│   ├── raw/...                   # MRI scan dataset (not included — see below)
│   └── experiments/              # Saved experiment outputs and metrics (generate from experiments)
│
├── presentation/                 # Final presentation slides
│
├── src/
│   ├── experiments/              # All experiments used in the dissertation
│   │
│   ├── models/                   # Early standalone model scripts from pilot phase
│   │                             # Note: not relevant to the final write-up and
│   │                             # incompatible with updated shared scripts
│   │
│   └── scripts/                  # Shared utilities used across all experiments
|
├── .gitignore                    # Repo/enviroment ignore file
├── status_report/                # Status report
├── timesheet.md                  # Project time log (minutes)
└── plan.md                       # Week-by-week project plan
```


> **Note on the data directory**: The `data/` directory is empty in this
> submission. Weight files (`.pt`) could not be included due to zip size
> constraints, and CSV result files have been omitted as they are
> generated as a by-product of training — without the weights they do not
> constitute a reproducible state. All contents are populated automatically
> by running the notebooks in order.


> **Note:** The standalone scripts in `src/models/` are early pilot implementations and are no longer compatible with the updated shared scripts. They are retained for reference only and are not required to reproduce any results in the dissertation.