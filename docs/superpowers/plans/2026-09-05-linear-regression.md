# Linear Regression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement ordinary linear regression from scratch (mini-batch gradient descent, plain numpy) and compare it against scikit-learn's own `LinearRegression` on the real Ames data.

**Architecture:** `build_pipeline()` gets a 4th step (`StandardScaler`) so gradient descent sees comparable feature scales. A new `LinearRegressionGD` class (plain numpy, sklearn-shaped `fit`/`predict` API, no `BaseEstimator` inheritance — it's the from-scratch part, not a pipeline transformer) implements vectorized mini-batch OLS. A comparison notebook fits both models on the same `log_sale_price` target and holdout split, reporting RMSE/R² and the GD loss curve.

**Tech Stack:** Python 3.13.15, numpy, pandas, scikit-learn, matplotlib, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-09-05-linear-regression-design.md`

## Global Constraints

- Python 3.13.15 (`pyproject.toml` `requires-python`); every command below runs via `uv run`.
- ruff: line-length 80, double quotes, rules `E,F,I,UP,B,SIM,RUF,S,PTH,C901,PLR0913` — `uv run ruff format <file>` then `uv run ruff check <file>` must be clean before each commit.
- mypy: `files = ["src"]`, `check_untyped_defs = true` — `uv run mypy src` must be clean; annotate `fit`/`predict` signatures.
- `LinearRegressionGD` is a plain class, **not** a `sklearn.base.BaseEstimator`/`TransformerMixin` subclass — per the spec's explicit "from scratch" scope, unlike `Imputer`/`FeatureEngineer`/`Encoder`. Its `fit`/`predict` types are `np.ndarray`, not `pd.DataFrame`.
- No regularization, no tolerance-based early stopping, no closed-form fallback — plain mini-batch OLS via gradient descent, exactly as decided in the spec.
- No comments explaining *what* code does — only *why*, when non-obvious. Match the rest of the repo.
- Model unit tests use small synthetic numpy arrays with a known linear relationship, not the real CSVs — only the notebook (Task 3) touches real data.

---

## Task 1: Append `StandardScaler` to `build_pipeline()`

**Files:**
- Modify: `src/ames_price/pipeline.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: nothing new — same `Imputer`/`FeatureEngineer`/`Encoder` composition already in `build_pipeline()`.
- Produces: `build_pipeline()`'s output is now standardized (`mean≈0`, `std≈1` per column). Task 3's notebook relies on this so `LinearRegressionGD` converges well.

- [ ] **Step 1: Update the failing/changed tests**

Replace the two `GarageAge` range assertions in `tests/test_pipeline.py` (they assert on raw-domain values, which no longer hold once the output is standardized) and add a new test for the standardization itself. Replace the whole file with:

```python
# tests/test_pipeline.py
import numpy as np
import pandas as pd

from ames_price.constants import NOMINAL_COLS, NUMERIC_COLS, ORDINAL_COLS
from ames_price.pipeline import build_pipeline

FEATURE_COLS = NUMERIC_COLS + ORDINAL_COLS + NOMINAL_COLS


def test_pipeline_produces_a_fully_numeric_matrix_with_no_nan():
    df_train = pd.read_csv("data/train.csv")
    df_test = pd.read_csv("data/test.csv")

    pipeline = build_pipeline()
    train_out = pipeline.fit_transform(df_train[FEATURE_COLS])
    test_out = pipeline.transform(df_test[FEATURE_COLS])

    assert not train_out.isna().any().any()
    assert not test_out.isna().any().any()
    assert train_out.shape[0] == len(df_train)
    assert test_out.shape[0] == len(df_test)
    assert train_out.shape[1] == 237
    assert list(train_out.columns) == list(test_out.columns)
    assert all(np.issubdtype(dtype, np.number) for dtype in train_out.dtypes)
    assert np.isfinite(train_out["GarageAge"]).all()
    assert np.isfinite(test_out["GarageAge"]).all()


def test_pipeline_output_is_standardized():
    df_train = pd.read_csv("data/train.csv")
    df_train = df_train[~df_train["Id"].isin([524, 1299])]

    pipeline = build_pipeline()
    train_out = pipeline.fit_transform(df_train[FEATURE_COLS])

    means = train_out.mean(axis=0)
    stds = train_out.std(axis=0, ddof=0)
    assert np.allclose(means, 0, atol=1e-6)
    # a zero-variance column (none expected, but not guaranteed by this
    # test) would legitimately stay at std 0 rather than 1 -- StandardScaler
    # skips dividing by a zero variance instead of raising or emitting NaN.
    assert np.all(
        np.isclose(stds, 1, atol=1e-6) | np.isclose(stds, 0, atol=1e-6)
    )
```

- [ ] **Step 2: Run tests to verify the new one fails**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: `test_pipeline_output_is_standardized` FAILS (mean/std not yet 0/1 — no scaler in the pipeline yet); the other test still PASSES (its assertions were loosened, not tightened).

- [ ] **Step 3: Add the scaler step**

In `src/ames_price/pipeline.py`, replace the whole file with:

```python
# src/ames_price/pipeline.py
from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ames_price.features import FeatureEngineer
from ames_price.preprocessing import Encoder, Imputer


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("impute", Imputer()),
            ("engineer", FeatureEngineer()),
            ("encode", Encoder()),
            ("scale", StandardScaler().set_output(transform="pandas")),
        ]
    )
```

`.set_output(transform="pandas")` keeps the pipeline's output a `pd.DataFrame` (with column names) instead of a bare `np.ndarray` — otherwise `test_pipeline_produces_a_fully_numeric_matrix_with_no_nan`'s `.isna()` calls would break, since numpy arrays have no such method.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Lint, format, typecheck**

Run: `uv run ruff format src/ames_price/pipeline.py tests/test_pipeline.py && uv run ruff check src/ames_price/pipeline.py tests/test_pipeline.py && uv run mypy src`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/ames_price/pipeline.py tests/test_pipeline.py
git commit -m "feat: append StandardScaler to build_pipeline()"
```

---

## Task 2: `LinearRegressionGD` — vectorized mini-batch gradient descent

**Files:**
- Create: `src/ames_price/models/__init__.py`
- Create: `src/ames_price/models/linear_regression.py`
- Test: `tests/test_linear_regression.py`

**Interfaces:**
- Consumes: nothing from earlier tasks — pure numpy, independent of the pipeline.
- Produces: `class LinearRegressionGD` with `__init__(self, learning_rate: float = 0.01, batch_size: int = 32, n_epochs: int = 100, random_state: int | None = None)`, `fit(self, X: np.ndarray, y: np.ndarray) -> LinearRegressionGD` (sets `coef_: np.ndarray`, `intercept_: float`, `loss_history_: list[float]`), `predict(self, X: np.ndarray) -> np.ndarray`. Task 3's notebook imports and calls this exactly as it calls sklearn's own `LinearRegression`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_linear_regression.py
import numpy as np

from ames_price.models.linear_regression import LinearRegressionGD


def _linear_fixture(
    n: int = 200, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    true_coef = np.array([3.0, -2.0])
    true_intercept = 5.0
    noise = rng.normal(scale=0.01, size=n)
    y = X @ true_coef + true_intercept + noise
    return X, y, true_coef, true_intercept


def test_fit_recovers_known_coefficients():
    X, y, true_coef, true_intercept = _linear_fixture()
    model = LinearRegressionGD(
        learning_rate=0.1, batch_size=20, n_epochs=300, random_state=42
    )
    model.fit(X, y)

    assert np.allclose(model.coef_, true_coef, atol=0.1)
    assert abs(model.intercept_ - true_intercept) < 0.1


def test_predict_output_shape_matches_input_rows():
    X, y, _, _ = _linear_fixture()
    model = LinearRegressionGD(n_epochs=10, random_state=42)
    model.fit(X, y)

    X_new = np.random.default_rng(0).normal(size=(5, 2))
    predictions = model.predict(X_new)

    assert predictions.shape == (5,)


def test_loss_history_length_and_non_increasing():
    X, y, _, _ = _linear_fixture()
    model = LinearRegressionGD(
        learning_rate=0.1, batch_size=len(X), n_epochs=200, random_state=42
    )
    model.fit(X, y)

    assert len(model.loss_history_) == 200
    assert all(
        model.loss_history_[i + 1] <= model.loss_history_[i] + 1e-9
        for i in range(len(model.loss_history_) - 1)
    )
```

`test_loss_history_length_and_non_increasing` sets `batch_size=len(X)` (full-batch GD) deliberately — mini-batch GD's per-*batch* updates aren't individually guaranteed to decrease the *full-training-set* loss, but full-batch GD at this `learning_rate` on this well-conditioned convex problem is (a small enough learning rate relative to the data's curvature guarantees non-increasing loss every step).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_linear_regression.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ames_price.models'`

- [ ] **Step 3: Write the implementation**

```python
# src/ames_price/models/__init__.py
```

(empty file — marks `models` as a package)

```python
# src/ames_price/models/linear_regression.py
from __future__ import annotations

import numpy as np


class LinearRegressionGD:
    """Ordinary linear regression fit by mini-batch gradient descent.

    Weights start at zero, the intercept is tracked as its own scalar
    (not folded into X via a ones-column) so each parameter's gradient
    update stays an explicit, separately-explainable line. Fixed epoch
    count, no tolerance-based early stopping -- loss_history_ is meant to
    be plotted and n_epochs picked by eye from where it flattens.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        n_epochs: int = 100,
        random_state: int | None = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)

        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0
        self.loss_history_: list[float] = []

        for _ in range(self.n_epochs):
            shuffled_idx = rng.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                batch_idx = shuffled_idx[start : start + self.batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]

                error = X_batch @ self.coef_ + self.intercept_ - y_batch
                grad_coef = (2 / len(batch_idx)) * (X_batch.T @ error)
                grad_intercept = (2 / len(batch_idx)) * error.sum()

                self.coef_ -= self.learning_rate * grad_coef
                self.intercept_ -= self.learning_rate * grad_intercept

            epoch_pred = X @ self.coef_ + self.intercept_
            epoch_loss = float(np.mean((epoch_pred - y) ** 2))
            self.loss_history_.append(epoch_loss)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_linear_regression.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Lint, format, typecheck**

Run: `uv run ruff format src/ames_price/models tests/test_linear_regression.py && uv run ruff check src/ames_price/models tests/test_linear_regression.py && uv run mypy src`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/ames_price/models tests/test_linear_regression.py
git commit -m "feat: add LinearRegressionGD (vectorized mini-batch gradient descent)"
```

---

## Task 3: `notebooks/models/1_linear_regression.ipynb`

**Files:**
- Create: `notebooks/models/1_linear_regression.ipynb`
- Create (temporary, scratchpad): a small Python script that builds the notebook JSON, matching the pattern used for `notebooks/preprocessing_check.ipynb`

**Interfaces:**
- Consumes: `build_pipeline()` from `ames_price.pipeline` (Task 1); `LinearRegressionGD` from `ames_price.models.linear_regression` (Task 2); `NUMERIC_COLS`/`ORDINAL_COLS`/`NOMINAL_COLS` from `ames_price.constants`.
- Produces: an executed, committed notebook — no code elsewhere depends on it.

- [ ] **Step 1: Write the notebook-build script**

```python
# scratchpad script (not committed) -- builds notebooks/models/1_linear_regression.ipynb
import json


def md(src):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


cells = [
    md(
        "# Linear regression: from scratch vs. scikit-learn\n\n"
        "Fits `LinearRegressionGD` (mini-batch gradient descent, plain "
        "numpy) and scikit-learn's own `LinearRegression` on the same "
        "preprocessed data and target, and compares them -- per "
        "`docs/superpowers/specs/2026-09-05-linear-regression-design.md`."
    ),
    code(
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n\n"
        "from sklearn.linear_model import LinearRegression\n"
        "from sklearn.metrics import mean_squared_error, r2_score\n"
        "from sklearn.model_selection import train_test_split\n\n"
        "from ames_price.constants import NOMINAL_COLS, NUMERIC_COLS, ORDINAL_COLS\n"
        "from ames_price.models.linear_regression import LinearRegressionGD\n"
        "from ames_price.pipeline import build_pipeline\n\n"
        "FEATURE_COLS = NUMERIC_COLS + ORDINAL_COLS + NOMINAL_COLS"
    ),
    md(
        "## Load data, drop outliers, build the target\n\n"
        "`Id` 524/1299 are the two outliers flagged in `6_outliers.ipynb`. "
        "The target is `log_sale_price` (`np.log1p(SalePrice)`, decided in "
        "`1_target_variable.ipynb`) -- both models train and are scored in "
        "log space, matching the actual Kaggle metric."
    ),
    code(
        'df_train = pd.read_csv("../../data/train.csv")\n'
        'df_train = df_train[~df_train["Id"].isin([524, 1299])]\n\n'
        "X = build_pipeline().fit_transform(df_train[FEATURE_COLS]).to_numpy()\n"
        'y = np.log1p(df_train["SalePrice"]).to_numpy()\n'
        "X.shape, y.shape"
    ),
    code(
        "X_train, X_val, y_train, y_val = train_test_split(\n"
        "    X, y, test_size=0.2, random_state=42\n"
        ")\n"
        "X_train.shape, X_val.shape"
    ),
    md("## Fit both models on the same training split"),
    code(
        "gd_model = LinearRegressionGD(\n"
        "    learning_rate=0.1, batch_size=32, n_epochs=300, random_state=42\n"
        ")\n"
        "gd_model.fit(X_train, y_train)\n\n"
        "sk_model = LinearRegression()\n"
        "sk_model.fit(X_train, y_train)"
    ),
    md(
        "## Compare on the holdout split\n\n"
        "RMSE in log space is the metric that actually matters (matches "
        "the Kaggle scoring); RMSE in dollar space (via `expm1`) is only "
        "there for human-readable intuition."
    ),
    code(
        "def rmse(y_true, y_pred):\n"
        "    return np.sqrt(mean_squared_error(y_true, y_pred))\n\n\n"
        'for name, model in [("LinearRegressionGD", gd_model), ("sklearn LinearRegression", sk_model)]:\n'
        "    preds_log = model.predict(X_val)\n"
        "    preds_dollars = np.expm1(preds_log)\n"
        "    actual_dollars = np.expm1(y_val)\n"
        "    print(name)\n"
        '    print("  RMSE (log space):", rmse(y_val, preds_log))\n'
        '    print("  RMSE (dollars, intuition only):", rmse(actual_dollars, preds_dollars))\n'
        '    print("  R^2:", r2_score(y_val, preds_log))'
    ),
    md(
        "## Convergence curve\n\n"
        "`loss_history_` is the per-epoch training MSE (log space) -- "
        "this is the actual stopping criterion for `n_epochs`: pick where "
        "this flattens, no automated tolerance check."
    ),
    code(
        "plt.plot(gd_model.loss_history_)\n"
        'plt.xlabel("epoch")\n'
        'plt.ylabel("training MSE (log_sale_price)")\n'
        'plt.title("LinearRegressionGD convergence")\n'
        "plt.show()"
    ),
    md(
        "## Coefficient comparison\n\n"
        "How close does gradient descent get to scikit-learn's "
        "closed-form (SVD-based least-squares) solution?"
    ),
    code(
        "coef_diff = np.abs(gd_model.coef_ - sk_model.coef_.ravel())\n"
        'print("max abs coefficient difference:", coef_diff.max())\n'
        'print("intercept -- GD:", gd_model.intercept_, " sklearn:", sk_model.intercept_)'
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.13.15"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
with open("notebooks/models/1_linear_regression.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
```

- [ ] **Step 2: Create the directory, run the script, execute the notebook**

Run:
```bash
mkdir -p notebooks/models
python3 /path/to/scratchpad/build_linear_regression_nb.py
cd notebooks/models && uv run --project ../.. jupyter nbconvert --to notebook --execute --inplace 1_linear_regression.ipynb
```
Expected: `[NbConvertApp] Writing ... bytes to 1_linear_regression.ipynb`, no error cells.

- [ ] **Step 3: Verify no errors**

Run:
```python
import json

nb = json.load(open("notebooks/models/1_linear_regression.ipynb"))
errs = [
    o
    for c in nb["cells"]
    if c.get("cell_type") == "code"
    for o in c.get("outputs", [])
    if o.get("output_type") == "error"
]
print("errors:", len(errs))
```
Expected: `errors: 0`

- [ ] **Step 4: Sanity-check the printed metrics**

Open the executed notebook and confirm: both models' log-space RMSE are close to each other (gradient descent should land within a small margin of sklearn's closed-form solution, not orders of magnitude off) and `max abs coefficient difference` is a small number, not huge — a large gap would mean the GD hyperparameters need adjusting (e.g. more epochs, different `learning_rate`) before this task is considered done.

- [ ] **Step 5: Commit**

```bash
git add notebooks/models/1_linear_regression.ipynb
git commit -m "feat: add notebook comparing LinearRegressionGD against sklearn's LinearRegression"
```

---

## Self-Review Notes

- **Spec coverage:** scaling step (Task 1), `LinearRegressionGD` — zero-init weights, separate intercept, vectorized mini-batch gradient, `loss_history_`, no regularization, no early stopping (Task 2), synthetic-data unit tests (Task 2), real-data comparison notebook with `log_sale_price` target, holdout split, RMSE (log + dollar)/R², loss-curve plot, coefficient comparison (Task 3) — every component in the spec has a task.
- **Type consistency checked:** `LinearRegressionGD.fit`/`predict` types (`np.ndarray` in, `np.ndarray`/`float` attributes out) are used consistently between Task 2's tests and Task 3's notebook; `build_pipeline()`'s output is a `pd.DataFrame` (via `.set_output(transform="pandas")`) which Task 3 explicitly converts with `.to_numpy()` before handing it to either model — avoiding a real bug where positional batch-index arrays would otherwise be misapplied as column labels against a DataFrame.
- **No placeholders:** every step has real code, real hyperparameters, or a real command; no "tune as needed" steps without concrete starting values.
