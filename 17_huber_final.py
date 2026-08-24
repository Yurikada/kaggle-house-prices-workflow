# -*- coding: utf-8 -*-
"""D（Huber）の確定測定。端追い探索（14/15/16）で判明した最適域を
両側から挟む統合gridを、内側CVに一度に見せる。E014と同じ設計。

併せて全学習データで学習し、test Id=2550 への予測と GrLivArea 係数を出す。
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
test = pd.read_csv(BASE / "test.csv")

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

# 端追い3回の結果、最適域は alpha≈300・epsilon 2〜3。両側から挟む。
GRID = {"model__alpha": [30.0, 100.0, 300.0, 1000.0, 3000.0],
        "model__epsilon": [1.35, 2.0, 3.0, 5.0]}

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
        va_ids = train.iloc[va_idx][ID].to_numpy()
        records.append({
            "variant": "D_huber_keep_all", "model": "Huber",
            "outer_seed": seed, "outer_fold": fold, "n_train": len(tr_idx),
            "rmse": root_mean_squared_error(y.iloc[va_idx], pred),
            "rmse_excl_1299": root_mean_squared_error(
                y.iloc[va_idx][va_ids != 1299], pred[va_ids != 1299]
            ) if (va_ids != 1299).sum() else np.nan,
            "best_alpha": search.best_params_["model__alpha"],
            "best_epsilon": search.best_params_["model__epsilon"],
        })
    print(f"  seed {seed} done ({time.perf_counter()-t0:.0f}s)", flush=True)

f = pd.DataFrame(records)
f.to_csv(Path(__file__).parent / "experiments" / "huber_final_folds.csv", index=False)
f["stratum"] = np.where(
    [(s, d) in HARD_FOLDS for s, d in zip(f.outer_seed, f.outer_fold)],
    "難fold3", "残り12")

print("\n=== D final (統合grid・単一測定) ===")
print(f"overall mean: {f.rmse.mean():.6f}")
print(f.groupby('stratum').rmse.mean().round(6).to_string())
print(f"excl_1299 mean: {f.rmse_excl_1299.mean():.6f}")
print("\n=== 内側CVが選んだ点（端が閉じているか）===")
print(f.groupby(['best_alpha', 'best_epsilon']).size().to_string())

print("\n=== 全学習データで学習 → test への予測 ===")
Xtest = b2(test)
search = GridSearchCV(pipe(X, HuberRegressor(max_iter=5000)), GRID,
                      scoring="neg_root_mean_squared_error",
                      cv=KFold(N_INNER, shuffle=True, random_state=7),
                      n_jobs=1, refit=True, error_score="raise")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)
    search.fit(X, y)
pt = np.expm1(search.predict(Xtest))
prep = search.best_estimator_.named_steps["prep"]
names = list(prep.get_feature_names_out())
coefs = np.asarray(search.best_estimator_.named_steps["model"].coef_).ravel()
print(f"best: {search.best_params_}")
for w in [2550, 2189, 2629]:
    print(f"Id={w}: {float(pt[test[ID] == w][0]):,.0f}")
print(f"test_max: {float(pt.max()):,.0f}")
print(f"coef_GrLivArea: {float(coefs[names.index('num__GrLivArea')]):.5f}")
print(f"\ntotal {time.perf_counter()-t0:.0f}s")

