# Fraud detection R&D (`fraud_rnd.ipynb`)

Catching fraud in the PaySim-style credit-card dataset (6.4M transactions, 31 days). The notebook builds features,
selects them, tunes a LightGBM model and a PLS + nearest-neighbour model, and validates both day by day against the
incumbent rule (check the largest transactions of at least $200k, up to 1000 per day).

## Environment and runtime:
I developoed this in a 96 core ec2 instance. I parallelized several of the steps that made sense but still takes several hours to run.

## Files

| File | What it is |
|---|---|
| `fraud_rnd.ipynb` | The notebook. Run this. |
| `requirements.txt` | Python dependencies. |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Select that environment as the notebook kernel. The dataset is downloaded on first run by `kagglehub`
(`dylanmoraes/credit-card-fraud-dataset`), so the machine needs internet access, and possibly Kaggle credentials
if your `kagglehub` version asks for them. Use a machine with plenty of RAM: the feature builders work on millions
of rows.

## Running top to bottom

Use **Kernel > Restart & Run All**, or run the cells in order. The notebook is order-dependent, and the cell that
splits off the black box deletes `df`, so don't re-run that cell on its own. If you need to redo it, restart and run
from the top.

| Section | What happens | Notes |
|---|---|---|
| Setup, data load, assumptions | Imports, downloads the data, sets `LAG = 1` and `LABEL_LAG = 24` hours | |
| 1-2) EDA | Plots, then the train / black box split (last ~5% by step) | Writes `fraud_blackbox_holdout.parquet` |
| 3) Feature engineering | `static_features` and `temporal_features` (account-history counts, fraud history) | Heavy on memory |
| 4) Isolation forest | `outlier_features` | Used later as 4 extra model features |
| 5) Incumbent baseline | The rule the model has to beat | |
| 6-7) Feature selection | Random feature-subset simulations, violin plots | Reads the saved CSV; the simulation call is commented out |
| 8) Fraud weight | Sweep of 90 class weights; the best is 1.61 | |
| 9) LightGBM tuning | Random search over 8 hyperparameters | |
| 10) LightGBM walk-forward | One-day look-ahead validation vs the incumbent | About 2 minutes per day, roughly 30 minutes in all |
| 11) PLS + kNN | Tunes N components and k neighbours, then walk-forward | Roughly 15 minutes |
| Black box + dollar losses | Walk-forward over train and the black box, then missed-fraud dollars | Slowest cell, about an hour. Run it once, with no tuning changes afterwards |

To skip the slow cells while reading the notebook, run everything above the walk-forward cells.

This takes several hours to run top to bottom. Many parts are parrelelized on a large machine.
