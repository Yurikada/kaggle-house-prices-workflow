# -*- coding: utf-8 -*-
"""Id=1299 の扱い、未測定だった D（頑健損失）と E（交互作用）を測る。採用はしない。

E014 (12_outlier_options.py) と同一の公平比較:
  - 行の除外や特徴量の追加は**各外側foldの学習部分にだけ**適用する
  - 評価は常に元の validation 全行、全variantで同一の行
  - B2特徴量、seed 0/4/42 × 5-fold、内側3-fold、dense grid、Pipeline固定

D: 行を消さず HuberRegressor（L2正則化＋Huber損失）で影響を抑える。
   線形モデルの比較条件が変わる点は E014 の注記どおり。
E: 行を消さず GrLivArea × (SaleCondition==Partial) の交互作用列を足す。
   ElasticNet / Ridge は E014 と同じ dense grid。

検算: A_keep_all の fold(0,1) を再実行し、E014 の保存値と一致するか照合する。
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
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
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


def add_interaction(features, frame):
    """E: Partial×面積。交互作用列は数値なので中央値補完＋標準化に乗る。"""
    out = features.copy()
    is_partial = (frame["SaleCondition"] == "Partial").astype(float)
    out["GrLivArea_x_Partial"] = out["GrLivArea"] * is_partial.to_numpy()
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
X_int = add_interaction(X, train)   # E用。行は同じ、列が1本増えるだけ
y = np.log1p(train[TARGET])

ALPHAS = [0.0001, 0.0003, 0.001, 0.003, 0.01]
EN_GRID = {"model__alpha": ALPHAS,
           "model__l1_ratio": [0.2, 0.5, 0.8, 0.9, 0.95, 1.0]}
RIDGE_GRID = {"model__alpha": [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]}
HUBER_GRID = {"model__alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1],
              "model__epsilon": [1.1, 1.35, 2.0]}

# variant名 → (特徴量, モデル仕様dict)
VARIANTS = {
    "D_huber_keep_all": (X, {"Huber": (HuberRegressor(max_iter=2000), HUBER_GRID)}),
    "E_partial_area_int": (X_int, {
        "ElasticNet": (ElasticNet(max_iter=200_000), EN_GRID),
        "Ridge": (Ridge(), RIDGE_GRID),
    }),
}

# ---- 検算: E014 の fold(0,1) A_keep_all を再現できるか ----
prev = pd.read_csv(PREV)
cv0 = KFold(n_splits=N_OUTER, shuffle=True, random_state=0)
tr_idx, va_idx = next(iter(cv0.split(X)))
inner0 = KFold(n_splits=N_INNER, shuffle=True, random_state=10_000 + 0 * 10 + 1)
for mname, (model, grid) in {
    "ElasticNet": (ElasticNet(max_iter=200_000), EN_GRID),
    "Ridge": (Ridge(), RIDGE_GRID),
}.items():
    search = GridSearchCV(pipe(X, clone(model)), grid,
                          scoring="neg_root_mean_squared_error",
                          cv=inner0, n_jobs=1, refit=True, error_score="raise")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        search.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    rmse = root_mean_squared_error(y.iloc[va_idx], search.predict(X.iloc[va_idx]))
    ref = prev.loc[(prev.variant == "A_keep_all") & (prev.model == mname)
                   & (prev.outer_seed == 0) & (prev.outer_fold == 1), "rmse"].iloc[0]
    print(f"検算 A(0,1) {mname:<10} 再現 {rmse:.12f} / E014 {ref:.12f} "
          f"差 {abs(rmse-ref):.2e}")

# ---- 本測定 ----
records = []
t0 = time.perf_counter()
for seed in OUTER_SEEDS:
    cv = KFold(n_splits=N_OUTER, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X), start=1):
        inner = KFold(n_splits=N_INNER, shuffle=True,
                      random_state=10_000 + seed * 10 + fold)
        va_ids = train.iloc[va_idx][ID].to_numpy()
        for variant, (feats, specs) in VARIANTS.items():
            Xva, yva = feats.iloc[va_idx], y.iloc[va_idx]
            Xtr, ytr = feats.iloc[tr_idx], y.iloc[tr_idx]
            for mname, (model, grid) in specs.items():
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
f.to_csv(Path(__file__).parent / "experiments" / "outlier_options_de_folds.csv", index=False)

both = pd.concat([prev, f], ignore_index=True)
both["stratum"] = np.where(
    [(s, d) in HARD_FOLDS for s, d in zip(both.outer_seed, both.outer_fold)],
    "難fold3", "残り12")

print("\n=== 同一validation上でのCV（A/B/CはE014保存値、D/Eは今回測定）===")
print(both.pivot_table(index="variant", columns="model",
                       values="rmse", aggfunc="mean").round(6).to_string())

print("\n=== 層別 ===")
print(both.pivot_table(index=["stratum", "variant"], columns="model",
                       values="rmse", aggfunc="mean").round(6).to_string())

print("\n=== 1299 自身を評価から抜いた場合 ===")
print(both.pivot_table(index="variant", columns="model",
                       values="rmse_excl_1299", aggfunc="mean").round(6).to_string())

print("\n=== 全学習データで学習 → test Id=2550 ほかへの予測 ===")
watch = [2550, 2189, 2629]
rows = []
for variant, (feats, specs) in VARIANTS.items():
    if variant.startswith("E_"):
        Xtest = add_interaction(b2(test), test)
    else:
        Xtest = b2(test)
    for mname, (model, grid) in specs.items():
        search = GridSearchCV(pipe(feats, clone(model)), grid,
                              scoring="neg_root_mean_squared_error",
                              cv=KFold(N_INNER, shuffle=True, random_state=7),
                              n_jobs=1, refit=True, error_score="raise")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            search.fit(feats, y)
        pt = np.expm1(search.predict(Xtest))
        entry = {"variant": variant, "model": mname,
                 "best": str(search.best_params_)}
        for w in watch:
            entry[f"Id={w}"] = round(float(pt[test[ID] == w][0]), 0)
        entry["test_max"] = round(float(pt.max()), 0)
        prep = search.best_estimator_.named_steps["prep"]
        names = list(prep.get_feature_names_out())
        coefs = np.asarray(search.best_estimator_.named_steps["model"].coef_).ravel()
        entry["coef_GrLivArea"] = round(float(coefs[names.index("num__GrLivArea")]), 5)
        if "num__GrLivArea_x_Partial" in names:
            entry["coef_interaction"] = round(
                float(coefs[names.index("num__GrLivArea_x_Partial")]), 5)
        rows.append(entry)
print(pd.DataFrame(rows).to_string(index=False))
print(f"\ntotal {time.perf_counter()-t0:.0f}s")

