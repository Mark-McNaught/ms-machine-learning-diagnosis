# Data Directory

This directory is populated automatically as notebooks are run in order.
No manual setup is required beyond installing dependencies — all result
files, plots, training curves, and CSVs will be written to the correct
subdirectories on completion of each notebook.

See `src/experiments/` for the notebooks and the main `README.md` for
the full run order.

## Trained weights

Model weight files (`.pt`) are not included in this submission zip due to
file size constraints. They are available by cloning the project repository:
```
https://github.com/Mark-McNaught/ms-machine-learning-diagnosis.git
```

Once cloned, all weights will be present under `data/experiments/*/weights/`
and no notebooks need to be re-run to reproduce test evaluations.