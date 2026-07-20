# Kaggle House Prices - Advanced Regression Techniques

Source competition: <https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques>

## Layout

- `data/`: local Kaggle competition files. Do not commit or redistribute unless permitted.
- `submissions/`: generated submission CSV files.
- `experiments/`: optional experiment scripts after the baseline is understood.
- `error_analysis/`: residual/error summaries after a local validation run exists.

The durable learning log is in the Vault:

```text
C:\Users\inada\OneDrive\ドキュメント\KnowledgeBase\90_Projects\Kaggle\house-prices-advanced-regression-techniques\experiment_log.md
```

## Start Rule

Do not start feature engineering or model selection here until the EDA checkpoint
in the Vault agent context has been filled by the user.

## Current Status

- URL workflow started on 2026-07-20.
- Kaggle data was downloaded locally after competition rules were accepted.
- `01_eda.py` passed a smoke check for shape, columns, missing values, target,
  and submission format.
- Feature selection, model selection, and baseline submission are intentionally
  not started yet.

## Data

Expected local files after Kaggle download:

```text
data/
  train.csv
  test.csv
  sample_submission.csv
  data_description.txt
```

Use Kaggle's official download method and keep competition data out of Git.
