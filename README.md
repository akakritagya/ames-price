# ames-price

Predicting house sale prices on the [Kaggle Ames Housing
dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) —
a learning project, so the emphasis throughout is on explaining *why* each
step exists, not just producing a leaderboard score.

## Project layout

- `notebooks/eda/0-7` — exploratory data analysis, one notebook per step
  (load & orient → target variable → missingness → univariate → bivariate →
  multicollinearity → outliers → wrap-up). Each is observational only: no
  fills, transforms, or drops happen here, only findings.
- `src/ames_price/` — the preprocessing/feature-engineering pipeline and
  the models built on top of it:
  - `constants.py` — shared column groupings and documented ordinal orders
  - `preprocessing.py` — `Imputer` (fills every `NaN`) and `Encoder`
    (ordinal/log1p/one-hot encoding)
  - `features.py` — `FeatureEngineer` (structural-identity resolution,
    derived features, rare-level bucketing)
  - `pipeline.py` — `build_pipeline()`, composing the imputer, feature
    engineer, encoder, and a `StandardScaler` into one
    `sklearn.pipeline.Pipeline`
  - `models/` — from-scratch mini-batch gradient descent models:
    `LinearRegressionGD`, plus the regularized `RidgeGD`/`LassoGD`/
    `ElasticNetGD`
- `notebooks/preprocessing_check.ipynb` — runs the pipeline end to end on
  the real data and visually verifies the output
- `notebooks/models/1_linear_regression.ipynb` — fits `LinearRegressionGD`
  and scikit-learn's `LinearRegression` on the same preprocessed data and
  compares them
- `notebooks/models/2_regularized_regression.ipynb` — fits `RidgeGD`/
  `LassoGD` and scikit-learn's `Ridge`/`Lasso` on the same preprocessed
  data, alongside the notebook 1 baseline, and compares them
- `notebooks/models/3_elastic_net.ipynb` — fits `ElasticNetGD` and
  scikit-learn's `ElasticNet` and compares them, alongside the
  notebook 1/2 baselines
- `tests/` — unit tests per transformer rule, plus one integration test
  against the real train/test CSVs
- `docs/superpowers/specs/` and `docs/superpowers/plans/` — the design
  docs and implementation plans behind the pipeline and models

Data (`train.csv`/`test.csv`) isn't versioned — download it from the Kaggle
competition page and place it under `data/`.

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
