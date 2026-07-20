# %% Imports
from pathlib import Path

import pandas as pd


# %% Paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submissions"


# %% Load data
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv")


# %% Basic shape and columns
print("train:", train.shape)
print("test:", test.shape)
print("sample_submission:", sample_submission.shape)

print(train.head())
print(test.head())
print(sample_submission.head())

print(train.columns.tolist())
print(test.columns.tolist())


# %% Missing values
missing_summary = pd.DataFrame(
    {
        "train_missing": train.isna().sum(),
        "test_missing": test.isna().sum(),
    }
).sort_values(["train_missing", "test_missing"], ascending=False)

print(missing_summary.head(30))


# %% Target and submission format checks
target_column = "SalePrice"
id_column = "Id"

assert target_column in train.columns
assert id_column in train.columns
assert id_column in test.columns
assert sample_submission.columns.tolist() == [id_column, target_column]
assert len(sample_submission) == len(test)

print(train[target_column].describe())

