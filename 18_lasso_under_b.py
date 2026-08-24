# -*- coding: utf-8 -*-
"""B（学習時のみ2行除外）の下で Lasso を測る。9aea94f3 の証拠を3モデル揃える。

設計は E014/E016 と同一: 除外は各外側foldの学習側のみ、評価は全行、
B2特徴量、seed 0/4/42 × 5-fold、内側3-fold、dense grid（E012 の等密度）。
EN との ケース単位ペア差もその場で出す（agentviz.report.paired_diff と同じ定義）。
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
from sklearn.linear_model import Lasso
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

TARGET, ID = "SalePrice", "Id"
OUTER_SEEDS, N_OUTER, N_INNER = [0, 4, 42], 5, 3
HARD_FOLDS = {(0, 1), (4, 3), (42, 3)}

BASE = Path(__file__).parent / "data"
PREV = Path(__file__).parent / "experiments" / "outlier_options_folds_e014.csv"
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
DROP = set(train.loc[(train.GrLivArea > 4000)
                     & (train.SaleCondition == "Partial"), ID])
print(f"B除外 {len(DROP)} 行 {sorted(DROP)}")

GRID = {"model__alpha": [0.0001, 0.0003, 0.001, 0.003, 0.01]}

records = []
t0 = time.perf_counter()
for seed in OUTER_SEEDS:
    cv = KFold(n_splits=N_OUTER, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X), start=1):
        inner = KFold(n_splits=N_INNER, shuffle=True,
                      random_state=10_000 + seed * 10 + fold)
        va_ids = train.iloc[va_idx][ID].to_numpy()
        keep = ~train.iloc[tr_idx][ID].isin(DROP).to_numpy()
        Xtr, ytr = X.iloc[tr_idx][keep], y.iloc[tr_idx][keep]
        search = GridSearchCV(pipe(Xtr, Lasso(max_iter=200_000)), GRID,
                              scoring="neg_root_mean_squared_error",
                              cv=inner, n_jobs=1, refit=True,
                              error_score="raise")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            search.fit(Xtr, ytr)
        pred = search.predict(X.iloc[va_idx])
        coefs = np.asarray(
            search.best_estimator_.named_steps["model"].coef_).ravel()
        records.append({
            "variant": "B_drop2_partial", "model": "Lasso",
            "outer_seed": seed, "outer_fold": fold, "n_train": len(Xtr),
            "rmse": root_mean_squared_error(y.iloc[va_idx], pred),
            "rmse_excl_1299": root_mean_squared_error(
                y.iloc[va_idx][va_ids != 1299], pred[va_ids != 1299]
            ) if (va_ids != 1299).sum() else np.nan,
            "best_alpha": search.best_params_["model__alpha"],
            "nonzero": int((np.abs(coefs) > 1e-10).sum()),
        })
    print(f"  seed {seed} done ({time.perf_counter()-t0:.0f}s)", flush=True)

f = pd.DataFrame(records)
f.to_csv(Path(__file__).parent / "experiments" / "lasso_under_b_folds.csv", index=False)
f["stratum"] = np.where(
    [(s, d) in HARD_FOLDS for s, d in zip(f.outer_seed, f.outer_fold)],
    "難fold3", "残り12")

print("\n=== Lasso under B ===")
print(f"overall mean: {f.rmse.mean():.6f}")
print(f.groupby('stratum').rmse.mean().round(6).to_string())
print(f"excl_1299 mean: {f.rmse_excl_1299.mean():.6f}")
print(f"nonzero coef: min {f.nonzero.min()} / mean {f.nonzero.mean():.1f} / "
      f"max {f.nonzero.max()}")
print("\n=== 内側CVが選んだ alpha ===")
print(f.groupby('best_alpha').size().to_string())

# EN(B) とのケース単位ペア差（E014保存値を基準に、Lassoを候補として）
prev = pd.read_csv(PREV)
en = prev[(prev.variant == "B_drop2_partial") & (prev.model == "ElasticNet")]
merged = f.merge(en, on=["outer_seed", "outer_fold"], suffixes=("_lasso", "_en"))
diff = merged.rmse_lasso - merged.rmse_en   # 負ならLassoが良い
wins = int((diff < -1e-12).sum())
losses = int((diff > 1e-12).sum())
print("\n=== ペア差 EN(B) → Lasso(B)（負ならLassoが良い）===")
print(f"平均 {diff.mean():+.6f} / 中央値 {diff.median():+.6f} / "
      f"Lasso {wins}勝{losses}敗{15-wins-losses}分")

print("\n=== 全学習データ(B除外)で学習 → test への予測 ===")
Xf, yf = X[~train[ID].isin(DROP).to_numpy()], y[~train[ID].isin(DROP).to_numpy()]
search = GridSearchCV(pipe(Xf, Lasso(max_iter=200_000)), GRID,
                      scoring="neg_root_mean_squared_error",
                      cv=KFold(N_INNER, shuffle=True, random_state=7),
                      n_jobs=1, refit=True, error_score="raise")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)
    search.fit(Xf, yf)
Xtest = b2(test)
pt = np.expm1(search.predict(Xtest))
prep = search.best_estimator_.named_steps["prep"]
names = list(prep.get_feature_names_out())
coefs = np.asarray(search.best_estimator_.named_steps["model"].coef_).ravel()
print(f"best: {search.best_params_}")
print(f"Id=2550: {float(pt[test[ID] == 2550][0]):,.0f}")
print(f"test_max: {float(pt.max()):,.0f}")
print(f"coef_GrLivArea: {float(coefs[names.index('num__GrLivArea')]):.5f}")
print(f"nonzero coef (full train): {int((np.abs(coefs) > 1e-10).sum())}")
print(f"\ntotal {time.perf_counter()-t0:.0f}s")

