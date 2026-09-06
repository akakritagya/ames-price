# ames-price

Predicting house sale prices on the [Kaggle Ames Housing
dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) —
a learning project, so the emphasis throughout is on explaining *why* each
step exists, not just producing a leaderboard score.

## Project layout

```text
ames-price/
├── src/ames_price/
│   ├── constants.py       # shared column groupings, documented ordinal orders
│   ├── preprocessing.py   # Imputer (fills every NaN) and Encoder (ordinal/log1p/one-hot)
│   ├── features.py        # FeatureEngineer (derived features, rare-level bucketing)
│   ├── pipeline.py        # build_pipeline() -- composes imputer, features, encoder, scaler
│   ├── model_selection.py # kfold_split() -- from-scratch k-fold index splitting
│   └── models/            # from-scratch mini-batch gradient descent models
│       ├── linear_regression.py  # LinearRegressionGD
│       ├── ridge_gd.py           # RidgeGD (L2)
│       ├── lasso_gd.py           # LassoGD (L1)
│       └── elastic_net_gd.py     # ElasticNetGD (L1+L2)
├── notebooks/
│   ├── eda/                       # 0-7, load & orient -> ... -> wrap-up (observational only)
│   ├── preprocessing_check.ipynb  # runs the pipeline end to end, visually verifies output
│   ├── models/
│   │   ├── 1_linear_regression.ipynb       # LinearRegressionGD vs sklearn LinearRegression
│   │   ├── 2_regularized_regression.ipynb  # RidgeGD/LassoGD vs sklearn Ridge/Lasso
│   │   ├── 3_elastic_net.ipynb             # ElasticNetGD vs sklearn ElasticNet
│   │   ├── 4_cross_validation.ipynb        # 5-fold CV hyperparameter selection vs. notebooks 2/3's single split
│   │   └── 5_submission.ipynb              # refits the best model on full train data, writes data/submission.csv
│   └── kaggle/
│       └── full_pipeline_regression.ipynb  # self-contained EDA->preprocessing->modeling->submission walkthrough, for Kaggle
├── tests/              # unit tests per transformer rule, plus a real-CSV integration test
├── docs/superpowers/
│   ├── specs/          # design docs behind the pipeline and models
│   └── plans/          # implementation plans behind the pipeline and models
└── data/               # train.csv / test.csv (not versioned -- see below)
```

`notebooks/eda/` steps run in order: load & orient → target variable →
missingness → univariate → bivariate → multicollinearity → outliers →
wrap-up. Each is observational only — no fills, transforms, or drops
happen there, only findings.

Data (`train.csv`/`test.csv`) isn't versioned — download it from the Kaggle
competition page and place it under `data/`.

## Results

Best model: sklearn `Lasso` (alpha=0.003162) — RMSE 0.113 (log space), R² 0.924
on a held-out split; `LassoGD`/`ElasticNetGD` track it closely. 5-fold CV
(notebook 4) confirmed 4 of 6 regularized models' single-split alpha picks
exactly — only Ridge shifted toward more regularization.

Notebook 5 refits this Lasso on the full training set and writes
`data/submission.csv` in the Kaggle-required `Id,SalePrice` format. The
competition's evaluation metric is RMSLE, which training against
`log1p(SalePrice)` already directly optimizes. Uploading the file to the
leaderboard is a manual step tied to a Kaggle account — see the notebook's
last cell for both the web UI and CLI options.

`notebooks/kaggle/full_pipeline_regression.ipynb` compresses the whole
project — EDA, preprocessing/feature engineering, all four model
families, CV-based hyperparameter selection, and submission — into one
documented, runnable notebook (~35s end to end). It installs the
`ames_price` package straight from this repo's GitHub so it can also run
unmodified as a Kaggle Code notebook, and locates the competition CSVs
under either `/kaggle/input/...` or this repo's `data/`, whichever exists.

## Using the pipeline

```python
import pandas as pd
from ames_price.constants import NOMINAL_COLS, NUMERIC_COLS, ORDINAL_COLS
from ames_price.pipeline import build_pipeline

df_train = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")

# drop the two flagged outliers before fitting anything (see 6_outliers.ipynb)
df_train = df_train[~df_train["Id"].isin([524, 1299])]

feature_cols = NUMERIC_COLS + ORDINAL_COLS + NOMINAL_COLS
pipeline = build_pipeline()
train_matrix = pipeline.fit_transform(df_train[feature_cols])
test_matrix = pipeline.transform(df_test[feature_cols])
```

## Development setup

```shell
git clone https://github.com/akakritagya/ames-price.git
cd ames-price
uv sync
uv run pre-commit install --install-hooks \
  --hook-type pre-commit --hook-type pre-push --hook-type commit-msg
```

## Verify setup

```shell
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```
