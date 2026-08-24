# -*- coding: utf-8 -*-
"""採用構成（D002/D003）で提出ファイルを作る。

構成: B2特徴量＋学習時のみ Id=524/1299 を除外（GrLivArea>4000 かつ Partial）
      ＋ ElasticNet（内側3-fold CVで alpha / l1_ratio を選択、dense grid）。
CV実測: 0.136481（seed 0/4/42 × 5-fold、除外は学習側のみ・評価は全行、E014）。

検算: 全学習fitは E016 と同一条件（random_state=7）なので、
      Id=2550 の予測 1,728,747 と best {'alpha': 0.0003, 'l1_ratio': 1.0} が
      再現されるはず。ずれたら環境差なので止まる。
"""
from pathlib import Path
import io
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

TARGET, ID = "SalePrice", "Id"
BASE = Path(__file__).parent
train = pd.read_csv(BASE / "data" / "train.csv")
test = pd.read_csv(BASE / "data" / "test.csv")

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

# D002: 学習時のみ2行除外
drop = (train.GrLivArea > 4000) & (train.SaleCondition == "Partial")
assert sorted(train.loc[drop, ID]) == [524, 1299]
Xf, yf = X[~drop.to_numpy()], y[~drop.to_numpy()]
print(f"学習行数: {len(Xf)}（除外 {sorted(train.loc[drop, ID])}）")

GRID = {"model__alpha": [0.0001, 0.0003, 0.001, 0.003, 0.01],
        "model__l1_ratio": [0.2, 0.5, 0.8, 0.9, 0.95, 1.0]}
search = GridSearchCV(pipe(Xf, ElasticNet(max_iter=200_000)), GRID,
                      scoring="neg_root_mean_squared_error",
                      cv=KFold(3, shuffle=True, random_state=7),
                      n_jobs=1, refit=True, error_score="raise")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)
    search.fit(Xf, yf)
print(f"best: {search.best_params_}")

pred = np.expm1(search.predict(b2(test)))
submission = pd.DataFrame({ID: test[ID], TARGET: pred})

# 検証（過去の提出検証と同じ観点）
assert submission.shape == (1459, 2)
assert (submission[ID].to_numpy() == test[ID].to_numpy()).all()
assert submission[TARGET].notna().all() and np.isfinite(pred).all()
assert (pred > 0).all()
id2550 = float(pred[test[ID] == 2550][0])
print(f"Id=2550 予測: {id2550:,.0f}（E016 実測 1,728,747 との検算）")
print(f"予測範囲: {pred.min():,.2f} 〜 {pred.max():,.2f}")

out = BASE / "submissions" / "submission_b_drop2_elasticnet.csv"
out.parent.mkdir(exist_ok=True)
submission.to_csv(out, index=False)
print(f"-> {out}")
