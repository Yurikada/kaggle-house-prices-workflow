# -*- coding: utf-8 -*-
"""D（Huber）のgrid端を閉じる。13_outlier_options_de.py で内側CVが
alpha=0.1・epsilon=2.0 と両方とも端を選んだため、上側へ広げて再測定する。

epsilon→∞ で Huber は二乗損失（＝Ridge相当）へ収束するので、
広げた先で Ridge A_keep_all (0.143619) に近づくかも同時に見える。
"""
from pathlib import Path
import io
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

TARGET, ID = "SalePrice", "Id"
OUTER_SEEDS, N_OUTER, N_INNER = [0, 4, 42], 5, 3
HARD_FOLDS = {(0, 1), (4, 3), (42, 3)}

BASE = Path(__file__).parent / "data"
train = pd.read_csv(BASE / "train.csv")

QUALITY = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
           "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond"]
QMAP = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}


def b2(frame):
    out = frame.drop(columns=[TARGET, ID], errors="ignore").copy()
    for c in QUALITY:
        out[c] = out[c].map(QMAP).fillna(0).astype(float)
    return out


def pipe(features, model):
    num = features.select_dtypes(include="number").columns.tolist()
    cat = features.select_dtypes(exclude="number").columns.tolist()
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="constant",
                                                    fill_value="Missing",
                                                    keep_empty_features=True)),
                              ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
        ])),
        ("model", model),
    ])


X = b2(train)
y = np.log1p(train[TARGET])

# 前回の端(0.1 / 2.0)を含み、上側へ延ばす
GRID = {"model__alpha": [3.0, 10.0, 30.0, 100.0, 300.0],
        "model__epsilon": [2.0, 3.0]}

records = []
t0 = time.perf_counter()
for seed in OUTER_SEEDS:
    cv = KFold(n_splits=N_OUTER, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X), start=1):
        inner = KFold(n_splits=N_INNER, shuffle=True,
                      random_state=10_000 + seed * 10 + fold)
        search = GridSearchCV(pipe(X, HuberRegressor(max_iter=5000)), GRID,
                              scoring="neg_root_mean_squared_error",
                              cv=inner, n_jobs=1, refit=True,
                              error_score="raise")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            search.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = search.predict(X.iloc[va_idx])
        records.append({
            "outer_seed": seed, "outer_fold": fold,
            "rmse": root_mean_squared_error(y.iloc[va_idx], pred),
            "best_alpha": search.best_params_["model__alpha"],
            "best_epsilon": search.best_params_["model__epsilon"],
        })
    print(f"  seed {seed} done ({time.perf_counter()-t0:.0f}s)", flush=True)

f = pd.DataFrame(records)
f.to_csv(Path(__file__).parent / "experiments" / "huber_widen2_folds.csv", index=False)
f["stratum"] = np.where(
    [(s, d) in HARD_FOLDS for s, d in zip(f.outer_seed, f.outer_fold)],
    "難fold3", "残り12")

print("\n=== D widened: 全体・層別 ===")
print(f"overall mean: {f.rmse.mean():.6f}")
print(f.groupby('stratum').rmse.mean().round(6).to_string())
print("\n=== 内側CVが選んだ点 ===")
print(f.groupby(['best_alpha', 'best_epsilon']).size().to_string())
print(f"\ntotal {time.perf_counter()-t0:.0f}s")


