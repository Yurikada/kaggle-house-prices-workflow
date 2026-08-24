# -*- coding: utf-8 -*-
"""Id=1299 の扱い A/B/C が実際に何をするかを測る。採用はしない。

公平性の要: 除外は**各外側foldの学習部分にだけ**適用し、評価は常に元の
validation全行で行う。除外行を評価からも消すとCVが自動的に良くなり、
比較にならないため。

測るもの:
  1. 同一validation上でのnested-CV（全体 / 難fold3 / 残り12）
  2. 全学習データで学習したときの test Id=2550 への予測
  3. GrLivArea 係数が除外でどう動くか
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
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
pd.set_option("display.width", 250)

TARGET, ID = "SalePrice", "Id"
OUTER_SEEDS, N_OUTER, N_INNER = [0, 4, 42], 5, 3
HARD_FOLDS = {(0, 1), (4, 3), (42, 3)}   # Id=1299 を含む3 fold

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

# 除外規則。V002 の調査どおり、規則によって巻き込む行が変わる。
DROP_IDS = {
    "A_keep_all": set(),
    "B_drop2_partial": set(train.loc[(train.GrLivArea > 4000)
                                     & (train.SaleCondition == "Partial"), ID]),
    "C_drop4_area": set(train.loc[train.GrLivArea > 4000, ID]),
}
for name, ids in DROP_IDS.items():
    print(f"{name:<18} 除外 {len(ids)} 行 {sorted(ids)}")

ALPHAS = [0.0001, 0.0003, 0.001, 0.003, 0.01]
SPECS = {
    "Ridge": (Ridge(), {"model__alpha": [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]}),
    "ElasticNet": (ElasticNet(max_iter=200_000),
                   {"model__alpha": ALPHAS,
                    "model__l1_ratio": [0.2, 0.5, 0.8, 0.9, 0.95, 1.0]}),
}

records = []
t0 = time.perf_counter()
for seed in OUTER_SEEDS:
    cv = KFold(n_splits=N_OUTER, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X), start=1):
        inner = KFold(n_splits=N_INNER, shuffle=True,
                      random_state=10_000 + seed * 10 + fold)
        # validation は変えない。全variantが同一行で評価される。
        Xva, yva = X.iloc[va_idx], y.iloc[va_idx]
        va_ids = train.iloc[va_idx][ID].to_numpy()

        for variant, drop in DROP_IDS.items():
            keep = ~train.iloc[tr_idx][ID].isin(drop).to_numpy()
            Xtr, ytr = X.iloc[tr_idx][keep], y.iloc[tr_idx][keep]
            for mname, (model, grid) in SPECS.items():
                search = GridSearchCV(pipe(Xtr, clone(model)), grid,
                                      scoring="neg_root_mean_squared_error",
                                      cv=inner, n_jobs=1, refit=True,
                                      error_score="raise")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    search.fit(Xtr, ytr)
                pred = search.predict(Xva)
                records.append({
                    "variant": variant, "model": mname,
                    "outer_seed": seed, "outer_fold": fold,
                    "n_train": len(Xtr),
                    "rmse": root_mean_squared_error(yva, pred),
                    "rmse_excl_1299": root_mean_squared_error(
                        yva[va_ids != 1299], pred[va_ids != 1299]
                    ) if (va_ids != 1299).sum() else np.nan,
                })
    print(f"  seed {seed} done ({time.perf_counter()-t0:.0f}s)", flush=True)

f = pd.DataFrame(records)
f.to_csv(Path(__file__).parent / "experiments" / "outlier_options_folds.csv", index=False)

f["stratum"] = np.where(
    [(s, d) in HARD_FOLDS for s, d in zip(f.outer_seed, f.outer_fold)],
    "難fold3", "残り12")

print("\n=== 同一validation上でのCV（除外は学習側のみ）===")
overall = f.pivot_table(index="variant", columns="model", values="rmse", aggfunc="mean")
print(overall.round(6).to_string())

print("\n=== 層別 ===")
strat = f.pivot_table(index=["stratum", "variant"], columns="model",
                      values="rmse", aggfunc="mean")
print(strat.round(6).to_string())

print("\n=== 1299 自身を評価から抜いた場合（他の行への影響だけを見る）===")
ex = f.pivot_table(index="variant", columns="model",
                   values="rmse_excl_1299", aggfunc="mean")
print(ex.round(6).to_string())

print("\n=== 全学習データで学習 → test Id=2550 ほかへの予測 ===")
Xtest = b2(test)
watch = [2550, 2189, 2629]
rows = []
for variant, drop in DROP_IDS.items():
    keep = ~train[ID].isin(drop).to_numpy()
    Xf, yf = X[keep], y[keep]
    for mname, (model, grid) in SPECS.items():
        search = GridSearchCV(pipe(Xf, clone(model)), grid,
                              scoring="neg_root_mean_squared_error",
                              cv=KFold(N_INNER, shuffle=True, random_state=7),
                              n_jobs=1, refit=True, error_score="raise")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            search.fit(Xf, yf)
        pt = np.expm1(search.predict(Xtest))
        entry = {"variant": variant, "model": mname,
                 "best": str(search.best_params_)}
        for w in watch:
            entry[f"Id={w}"] = round(float(pt[test[ID] == w][0]), 0)
        entry["test_max"] = round(float(pt.max()), 0)
        # GrLivArea の係数（標準化後）
        prep = search.best_estimator_.named_steps["prep"]
        names = list(prep.get_feature_names_out())
        coefs = np.asarray(search.best_estimator_.named_steps["model"].coef_).ravel()
        gi = names.index("num__GrLivArea")
        entry["coef_GrLivArea"] = round(float(coefs[gi]), 5)
        rows.append(entry)
print(pd.DataFrame(rows).to_string(index=False))
print(f"\ntotal {time.perf_counter()-t0:.0f}s")

