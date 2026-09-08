# Kaggle House Prices — 外れ値処理とモデル選択の検証

[House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)を題材に、外れ値の扱いとモデル選択を、同じ検証条件で比較する学習プロジェクトです。

An independent learning case study of outlier handling and nested-CV model selection. The public snapshot includes EDA, later comparison scripts, fold-level results, and submission-file generation.

## 現在の公開版

EDAに加え、Huberの探索範囲、Lassoとの比較、採用したElasticNet構成の提出ファイル生成コードまで公開しています。`learning_log.md` は2026-07-20の開始時点の記録で、現在の実装範囲を示すものではありません。

- 採用構成: 品質列の順序尺度化を含むB2特徴量＋ElasticNet。
- 外れ値処理: 学習側のみ `Id=524/1299` を除外し、交差検証の評価側は全行を残す。
- 記録されたCV: log1p価格のRMSE `0.136481`（seed 0/4/42 × 外側5-fold）。これは公開コードに記録された実験値です。
- 学習済みモデルやコンペデータは同梱していません。

## 公開されている内容

| 入口 | 内容 |
|---|---|
| [01_eda.py](01_eda.py) | データ形状・欠損・目的変数・提出形式の確認 |
| [11_outlier_survey.py](11_outlier_survey.py) | 外れ値候補の調査 |
| [12_outlier_options.py](12_outlier_options.py) / [13_outlier_options_de.py](13_outlier_options_de.py) | 外れ値処理案の比較 |
| [14_huber_widen.py](14_huber_widen.py)〜[17_huber_final.py](17_huber_final.py) | Huberの探索範囲を広げ、比較条件を揃える |
| [18_lasso_under_b.py](18_lasso_under_b.py) | 学習側2行除外の条件でLassoを比較 |
| [experiments/](experiments/) | fold別の比較結果CSV |
| [19_create_submission_b.py](19_create_submission_b.py) | 採用構成で提出用CSVを生成し、行数・ID順・有限値・正値を検査 |

番号02〜10のスクリプトや学習Notebook一式は、この公開スナップショットには含まれていません。番号順に全履歴を再実行できる構成ではなく、各公開スクリプトの入力と依存を確認して使います。

## セットアップと実行

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python 01_eda.py
```

事前にKaggleの公式手段でデータを取得し、ZIPを次の構成へ展開してください。

```text
data/
  train.csv
  test.csv
  sample_submission.csv
  data_description.txt
```

採用構成のCSVを再生成する場合:

```bash
python 19_create_submission_b.py
```

出力は `submissions/submission_b_drop2_elasticnet.csv`。このコマンドはローカルファイルを生成します。Kaggleへの送信は行いません。

## 比較するときの注意

- 外れ値を除く条件と、誤差を測る母集団を区別します。
- 平均スコアと合わせてfold別結果を読みます。特定行を含むfoldで順位が変わることがあります。
- 依存バージョンは完全には固定されていません。収束状況や再生成値の差を確認してください。
- コンペデータ・個人の作業ログ・生成した提出CSVはGit管理外です。

実験比較の可視化には [agent-viz](https://github.com/Yurikada/agent-viz) を使用しています。

## License

MIT — see [LICENSE](LICENSE).
