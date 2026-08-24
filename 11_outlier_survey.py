# -*- coding: utf-8 -*-
"""Id=1299 の扱いを決めるための事実調査。除去も採用も行わない。

問い:
  - 1299/524 は何が特異なのか
  - 「こういう行」を指す規則を書くと train の何行が該当するのか
  - 同じ性質の行が test 側にも居るのか（居るなら train から消す判断の意味が変わる）
"""
from pathlib import Path
import io
import sys

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)

BASE = Path(__file__).parent / "data"
train = pd.read_csv(BASE / "train.csv")
test = pd.read_csv(BASE / "test.csv")

COLS = ["Id", "GrLivArea", "LotArea", "OverallQual", "OverallCond", "YearBuilt",
        "Neighborhood", "SaleCondition", "SaleType", "TotalBsmtSF", "SalePrice"]

print("=== 大誤差2行の素性 ===")
print(train.loc[train.Id.isin([1299, 524]), COLS].to_string(index=False))

print("\n=== GrLivArea 上位8行 ===")
top = train.nlargest(8, "GrLivArea")[COLS]
top["price_per_sf"] = (top.SalePrice / top.GrLivArea).round(1)
print(top.to_string(index=False))

print("\n=== GrLivArea>4000 の SaleCondition 内訳（train）===")
big = train[train.GrLivArea > 4000]
print(big.groupby("SaleCondition").agg(
    n=("Id", "size"),
    median_price=("SalePrice", "median"),
    median_sf=("GrLivArea", "median"),
).to_string())

print("\n=== 候補規則ごとの該当行数 ===")
rules = {
    "GrLivArea > 4000": train.GrLivArea > 4000,
    "GrLivArea > 4500": train.GrLivArea > 4500,
    "GrLivArea > 4000 かつ SaleCondition == Partial": (train.GrLivArea > 4000) & (train.SaleCondition == "Partial"),
    "GrLivArea > 4000 かつ SalePrice < 300000": (train.GrLivArea > 4000) & (train.SalePrice < 300_000),
    "SaleCondition == Partial（全体）": train.SaleCondition == "Partial",
}
for name, mask in rules.items():
    ids = train.loc[mask, "Id"].tolist()
    print(f"  {name:<46} {mask.sum():>4} 行  {'含1299' if 1299 in ids else '':<7}"
          f"{'含524' if 524 in ids else ''}")

print("\n=== test 側に同じ性質の行が居るか ===")
for name, expr in [
    ("GrLivArea > 4000", lambda d: d.GrLivArea > 4000),
    ("GrLivArea > 4500", lambda d: d.GrLivArea > 4500),
    ("SaleCondition == Partial", lambda d: d.SaleCondition == "Partial"),
]:
    print(f"  {name:<28} train {expr(train).sum():>5} 行   test {expr(test).sum():>5} 行")

print("\n=== test の GrLivArea 上位5行（予測を出す相手）===")
print(test.nlargest(5, "GrLivArea")[
    ["Id", "GrLivArea", "OverallQual", "Neighborhood", "SaleCondition", "YearBuilt"]
].to_string(index=False))

print("\n=== Partial 売却の価格水準（train）===")
g = train.groupby("SaleCondition").agg(
    n=("Id", "size"),
    median_price=("SalePrice", "median"),
    median_grlivarea=("GrLivArea", "median"),
).sort_values("n", ascending=False)
print(g.to_string())

print("\n=== 1299/524 は「安すぎる」のか ===")
# 同程度の広さ・品質の行と比べる
peers = train[(train.GrLivArea > 3000) & (train.OverallQual >= 9)]
print(f"GrLivArea>3000 かつ OverallQual>=9 の {len(peers)} 行:")
print(peers[["Id", "GrLivArea", "OverallQual", "Neighborhood", "SaleCondition", "SalePrice"]]
      .sort_values("SalePrice").to_string(index=False))

