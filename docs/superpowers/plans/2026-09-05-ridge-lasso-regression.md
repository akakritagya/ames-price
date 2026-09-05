# Ridge & Lasso Regression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Ridge (L2) and Lasso (L1) regularized linear regression from scratch (mini-batch gradient descent, plain numpy), and compare each against scikit-learn's own `Ridge`/`Lasso` on the real Ames data, alongside the existing plain-regression baseline.

**Architecture:** Two new estimator classes (`RidgeGD`, `LassoGD`), each in its own file, mirroring `LinearRegressionGD`'s exact skeleton (zero-init weights, separately tracked intercept, seeded mini-batch loop, `loss_history_`) plus one penalty term added to the coefficient gradient. A new notebook sweeps each model family's `alpha` independently against validation RMSE, refits at each family's best alpha, and reports metrics plus a coefficient-sparsity comparison against the existing plain-regression baseline.

**Tech Stack:** Python 3.13.15, numpy, pandas, scikit-learn, matplotlib, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-09-05-ridge-lasso-regression-design.md`

## Global Constraints

- Python 3.13.15 (`pyproject.toml` `requires-python`); every command below runs via `uv run`.
- ruff: line-length 80, double quotes, rules `E,F,I,UP,B,SIM,RUF,S,PTH,C901,PLR0913,D` (numpy pydocstyle convention) — `uv run ruff format <file>` then `uv run ruff check <file>` must be clean before each commit. Every public class/method needs a numpy-style docstring (`Parameters`/`Returns` sections), matching `src/ames_price/models/linear_regression.py`'s existing style.
- mypy: `files = ["src"]`, `check_untyped_defs = true` — `uv run mypy src` must be clean; annotate `fit`/`predict` signatures.
- `RidgeGD`/`LassoGD` are plain classes, **not** `sklearn.base.BaseEstimator`/`TransformerMixin` subclasses — same "from scratch" scope as `LinearRegressionGD`. Their `fit`/`predict` types are `np.ndarray`, not `pd.DataFrame`.
- Both penalize `coef_` only — the intercept is always updated by the plain unregularized gradient (`(2 / len(batch_idx)) * error.sum()`), exactly as in `LinearRegressionGD`.
- `RidgeGD`'s penalty gradient term is `2 * alpha * coef_` (not scaled by batch size); `LassoGD`'s is `alpha * np.sign(coef_)` (subgradient, `np.sign(0) == 0`). Neither uses a proximal/soft-thresholding solver.
- `loss_history_` on both records the full regularized objective (`MSE + alpha * penalty(coef_)`) per epoch, not bare MSE — document this explicitly in the docstring since it's not directly comparable to `LinearRegressionGD.loss_history_`.
- No ElasticNet, no coordinate descent, no cross-validation, no early stopping — fixed `n_epochs`, exactly as decided in the spec.
- No attempt to make GD's `alpha` numerically match sklearn's `alpha` — each model family's alpha sweep runs independently; comparison is at the level of best-achievable validation RMSE, not identical alpha meaning.
- No comments explaining *what* code does — only *why*, when non-obvious. Match the rest of the repo.
- Model unit tests use small synthetic numpy arrays with a known linear relationship, not the real CSVs — only the notebook (Task 3) touches real data.

---

## Task 1: `RidgeGD` — L2-regularized mini-batch gradient descent

**Files:**
- Create: `src/ames_price/models/ridge_gd.py`
- Test: `tests/test_ridge_gd.py`

**Interfaces:**
- Consumes: nothing from earlier work — pure numpy, independent of the pipeline. Same shape as `LinearRegressionGD` (`src/ames_price/models/linear_regression.py`), which this file sits alongside in `src/ames_price/models/`.
- Produces: `class RidgeGD` with `__init__(self, learning_rate: float = 0.01, batch_size: int = 32, n_epochs: int = 100, alpha: float = 1.0, random_state: int | None = None)`, `fit(self, X: np.ndarray, y: np.ndarray) -> RidgeGD` (sets `coef_: np.ndarray`, `intercept_: float`, `loss_history_: list[float]`), `predict(self, X: np.ndarray) -> np.ndarray`. Task 3's notebook imports and calls this exactly as it calls `LinearRegressionGD` and sklearn's own `Ridge`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ridge_gd.py
import numpy as np

from ames_price.models.ridge_gd import RidgeGD


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


def test_fit_recovers_approximately_correct_coefficients():
    X, y, true_coef, true_intercept = _linear_fixture()
    model = RidgeGD(
        learning_rate=0.1,
        batch_size=20,
        n_epochs=300,
        alpha=0.1,
        random_state=42,
    )
    model.fit(X, y)

    # Ridge is intentionally biased toward smaller |coef_| than the true
    # weights -- a looser tolerance than plain OLS's, not a bug.
    assert np.allclose(model.coef_, true_coef, atol=0.3)
    # the intercept is never penalized, so it still recovers tightly.
    assert abs(model.intercept_ - true_intercept) < 0.1


def test_predict_output_shape_matches_input_rows():
    X, y, _, _ = _linear_fixture()
    model = RidgeGD(n_epochs=10, random_state=42)
    model.fit(X, y)

    X_new = np.random.default_rng(0).normal(size=(5, 2))
    predictions = model.predict(X_new)

    assert predictions.shape == (5,)


def test_loss_history_length_and_non_increasing():
    X, y, _, _ = _linear_fixture()
    model = RidgeGD(
        learning_rate=0.1,
        batch_size=len(X),
        n_epochs=200,
        alpha=0.1,
        random_state=42,
    )
    model.fit(X, y)

    assert len(model.loss_history_) == 200
    assert all(
        model.loss_history_[i + 1] <= model.loss_history_[i] + 1e-9
        for i in range(len(model.loss_history_) - 1)
    )
```

`test_loss_history_length_and_non_increasing` uses `batch_size=len(X)` (full-batch GD) for the same reason as `LinearRegressionGD`'s equivalent test: mini-batch updates aren't individually guaranteed to decrease the full-training-set *regularized* objective, but full-batch GD at this `learning_rate` on this well-conditioned convex problem (the L2 penalty only makes the objective *more* convex, never less) is.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ridge_gd.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ames_price.models.ridge_gd'`

- [ ] **Step 3: Write the implementation**

```python
# src/ames_price/models/ridge_gd.py
"""Ridge-regularized linear regression fit by mini-batch gradient descent."""

from __future__ import annotations

import numpy as np


class RidgeGD:
    """Linear regression with an L2 penalty, fit by mini-batch gradient descent.

    Same skeleton as ``LinearRegressionGD``: weights start at zero, the
    intercept is tracked as its own scalar, and the batch loop is
    identical. The only difference is an ``alpha * sum(coef_ ** 2)``
    penalty added to the objective and its gradient -- excluded from the
    intercept's update, since only the relationship to features should
    shrink, not the model's baseline prediction.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        n_epochs: int = 100,
        alpha: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        """Store hyperparameters; fitting happens in fit(), not here.

        Parameters
        ----------
        learning_rate : float, default=0.01
            Step size applied to each gradient update.
        batch_size : int, default=32
            Number of rows per mini-batch. A value >= n_samples degrades
            to full-batch gradient descent.
        n_epochs : int, default=100
            Number of passes over the shuffled training data.
        alpha : float, default=1.0
            L2 penalty strength applied to coef_. The intercept is never
            penalized.
        random_state : int or None, default=None
            Seed for the per-epoch row shuffle. None gives a different,
            non-reproducible shuffle on every fit() call.
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> RidgeGD:
        """Fit by mini-batch gradient descent on the L2-penalized objective.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        RidgeGD
            The fitted estimator, with coef_, intercept_, and
            loss_history_ set.
        """
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
                grad_coef = (2 / len(batch_idx)) * (
                    X_batch.T @ error
                ) + 2 * self.alpha * self.coef_
                grad_intercept = (2 / len(batch_idx)) * error.sum()

                self.coef_ -= self.learning_rate * grad_coef
                self.intercept_ -= self.learning_rate * grad_intercept

            epoch_pred = X @ self.coef_ + self.intercept_
            epoch_mse = float(np.mean((epoch_pred - y) ** 2))
            epoch_penalty = self.alpha * float(np.sum(self.coef_**2))
            self.loss_history_.append(epoch_mse + epoch_penalty)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples to predict on.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted values, X @ coef_ + intercept_.
        """
        return X @ self.coef_ + self.intercept_
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ridge_gd.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Lint, format, typecheck**

Run: `uv run ruff format src/ames_price/models/ridge_gd.py tests/test_ridge_gd.py && uv run ruff check src/ames_price/models/ridge_gd.py tests/test_ridge_gd.py && uv run mypy src`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/ames_price/models/ridge_gd.py tests/test_ridge_gd.py
git commit -m "feat: add RidgeGD (L2-regularized mini-batch gradient descent)"
```

---

## Task 2: `LassoGD` — L1-regularized mini-batch gradient descent

**Files:**
- Create: `src/ames_price/models/lasso_gd.py`
- Test: `tests/test_lasso_gd.py`

**Interfaces:**
- Consumes: nothing from earlier tasks — pure numpy, structurally identical to `RidgeGD` (Task 1) and `LinearRegressionGD` except for the penalty term.
- Produces: `class LassoGD` with `__init__(self, learning_rate: float = 0.01, batch_size: int = 32, n_epochs: int = 100, alpha: float = 1.0, random_state: int | None = None)`, `fit(self, X: np.ndarray, y: np.ndarray) -> LassoGD` (sets `coef_: np.ndarray`, `intercept_: float`, `loss_history_: list[float]`), `predict(self, X: np.ndarray) -> np.ndarray`. Task 3's notebook imports and calls this exactly as it calls `RidgeGD`/`LinearRegressionGD` and sklearn's own `Lasso`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_lasso_gd.py
import numpy as np

from ames_price.models.lasso_gd import LassoGD


def _sparse_linear_fixture(
    n: int = 200, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    # x1, x2 are informative; x3, x4 have a true coefficient of exactly
    # zero -- this is what the sparsity assertion below checks for.
    true_coef = np.array([3.0, -2.0, 0.0, 0.0])
    true_intercept = 5.0
    noise = rng.normal(scale=0.01, size=n)
    y = X @ true_coef + true_intercept + noise
    return X, y, true_coef, true_intercept


def test_fit_shrinks_irrelevant_features_toward_zero():
    X, y, true_coef, true_intercept = _sparse_linear_fixture()
    model = LassoGD(
        learning_rate=0.1,
        batch_size=20,
        n_epochs=300,
        alpha=0.5,
        random_state=42,
    )
    model.fit(X, y)

    # the two true-zero features are pushed near zero -- Lasso's defining
    # behavior (feature selection), not just "it fits."
    assert abs(model.coef_[2]) < 0.1
    assert abs(model.coef_[3]) < 0.1
    # the informative features stay close to their true values -- L1's
    # constant-magnitude penalty biases these less than a naive reading
    # of "regularization shrinks everything" would suggest.
    assert np.allclose(model.coef_[:2], true_coef[:2], atol=0.5)
    assert abs(model.intercept_ - true_intercept) < 0.1


def test_predict_output_shape_matches_input_rows():
    X, y, _, _ = _sparse_linear_fixture()
    model = LassoGD(n_epochs=10, random_state=42)
    model.fit(X, y)

    X_new = np.random.default_rng(0).normal(size=(5, 4))
    predictions = model.predict(X_new)

    assert predictions.shape == (5,)


def test_loss_history_length_and_non_increasing():
    X, y, _, _ = _sparse_linear_fixture()
    model = LassoGD(
        learning_rate=0.01,
        batch_size=len(X),
        n_epochs=200,
        alpha=0.5,
        random_state=42,
    )
    model.fit(X, y)

    assert len(model.loss_history_) == 200
    assert all(
        model.loss_history_[i + 1] <= model.loss_history_[i] + 1e-9
        for i in range(len(model.loss_history_) - 1)
    )
```

`test_loss_history_length_and_non_increasing` uses a smaller `learning_rate` (0.01, vs. Ridge's 0.1) — the L1 subgradient's constant magnitude (not proportional to `coef_`, unlike L2) can otherwise cause a small oscillation right around convergence that would violate strict non-increase; full-batch GD at this smaller step size on this well-conditioned problem stays monotonically non-increasing.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_lasso_gd.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ames_price.models.lasso_gd'`

- [ ] **Step 3: Write the implementation**

```python
# src/ames_price/models/lasso_gd.py
"""Lasso-regularized linear regression fit by mini-batch gradient descent."""

from __future__ import annotations

import numpy as np


class LassoGD:
    """Linear regression with an L1 penalty, fit by mini-batch gradient descent.

    Same skeleton as ``LinearRegressionGD``/``RidgeGD``: weights start at
    zero, the intercept is tracked as its own scalar, and the batch loop
    is identical. The only difference is an ``alpha * sum(abs(coef_))``
    penalty term, excluded from the intercept's update.

    ``abs(coef_)`` has no gradient at coef_ == 0, so this uses the
    subgradient approximation ``alpha * np.sign(coef_)`` (with
    ``np.sign(0) == 0``) rather than a proximal/soft-thresholding solver
    -- simpler, and keeps this class's fit() structurally identical to
    RidgeGD's. Coefficients shrink *toward* zero and settle very close to
    it, but this does not produce the bitwise-exact zeros a proximal
    solver would -- check "near zero" with a tolerance, not equality.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        n_epochs: int = 100,
        alpha: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        """Store hyperparameters; fitting happens in fit(), not here.

        Parameters
        ----------
        learning_rate : float, default=0.01
            Step size applied to each gradient update.
        batch_size : int, default=32
            Number of rows per mini-batch. A value >= n_samples degrades
            to full-batch gradient descent.
        n_epochs : int, default=100
            Number of passes over the shuffled training data.
        alpha : float, default=1.0
            L1 penalty strength applied to coef_. The intercept is never
            penalized.
        random_state : int or None, default=None
            Seed for the per-epoch row shuffle. None gives a different,
            non-reproducible shuffle on every fit() call.
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> LassoGD:
        """Fit by mini-batch gradient descent on the L1-penalized objective.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        LassoGD
            The fitted estimator, with coef_, intercept_, and
            loss_history_ set.
        """
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
                grad_coef = (2 / len(batch_idx)) * (
                    X_batch.T @ error
                ) + self.alpha * np.sign(self.coef_)
                grad_intercept = (2 / len(batch_idx)) * error.sum()

                self.coef_ -= self.learning_rate * grad_coef
                self.intercept_ -= self.learning_rate * grad_intercept

            epoch_pred = X @ self.coef_ + self.intercept_
            epoch_mse = float(np.mean((epoch_pred - y) ** 2))
            epoch_penalty = self.alpha * float(np.sum(np.abs(self.coef_)))
            self.loss_history_.append(epoch_mse + epoch_penalty)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples to predict on.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted values, X @ coef_ + intercept_.
        """
        return X @ self.coef_ + self.intercept_
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_lasso_gd.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Lint, format, typecheck**

Run: `uv run ruff format src/ames_price/models/lasso_gd.py tests/test_lasso_gd.py && uv run ruff check src/ames_price/models/lasso_gd.py tests/test_lasso_gd.py && uv run mypy src`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/ames_price/models/lasso_gd.py tests/test_lasso_gd.py
git commit -m "feat: add LassoGD (L1-regularized mini-batch gradient descent)"
```

---

## Task 3: `notebooks/models/2_regularized_regression.ipynb`

**Files:**
- Create: `notebooks/models/2_regularized_regression.ipynb`
- Create (temporary, scratchpad): a small Python script that builds the notebook JSON, matching the pattern used for `notebooks/models/1_linear_regression.ipynb`

**Interfaces:**
- Consumes: `build_pipeline()` from `ames_price.pipeline`; `LinearRegressionGD` from `ames_price.models.linear_regression`; `RidgeGD` from `ames_price.models.ridge_gd` (Task 1); `LassoGD` from `ames_price.models.lasso_gd` (Task 2); `NUMERIC_COLS`/`ORDINAL_COLS`/`NOMINAL_COLS` from `ames_price.constants`; sklearn's `Ridge`, `Lasso`, `LinearRegression`.
- Produces: an executed, committed notebook — no code elsewhere depends on it.

- [ ] **Step 1: Write the notebook-build script**

```python
# scratchpad script (not committed) -- builds
# notebooks/models/2_regularized_regression.ipynb
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
        "# Regularized regression: Ridge & Lasso, from scratch vs. "
        "scikit-learn\n\n"
        "Fits `RidgeGD`/`LassoGD` (mini-batch gradient descent, plain "
        "numpy) and scikit-learn's own `Ridge`/`Lasso` on the same "
        "preprocessed data and target, alongside the plain-regression "
        "baseline from `1_linear_regression.ipynb` -- per "
        "`docs/superpowers/specs/2026-09-05-ridge-lasso-regression-design.md`."
    ),
    code(
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n\n"
        "from sklearn.linear_model import Lasso, LinearRegression, Ridge\n"
        "from sklearn.metrics import (\n"
        "    mean_absolute_error,\n"
        "    mean_absolute_percentage_error,\n"
        "    mean_squared_error,\n"
        "    r2_score,\n"
        ")\n"
        "from sklearn.model_selection import train_test_split\n\n"
        "from ames_price.constants import NOMINAL_COLS, NUMERIC_COLS, ORDINAL_COLS\n"
        "from ames_price.models.lasso_gd import LassoGD\n"
        "from ames_price.models.linear_regression import LinearRegressionGD\n"
        "from ames_price.models.ridge_gd import RidgeGD\n"
        "from ames_price.pipeline import build_pipeline\n\n"
        "FEATURE_COLS = NUMERIC_COLS + ORDINAL_COLS + NOMINAL_COLS"
    ),
    md(
        "## Load data, drop outliers, build the target, split before "
        "fitting\n\n"
        "`Id` 524/1299 are the two outliers flagged in `6_outliers.ipynb`. "
        "The target is `log_sale_price` (`np.log1p(SalePrice)`), same as "
        "notebook 1. The pipeline is fit on the *training split only* and "
        "applied to validation with `transform` -- fitting it on the full "
        "data before splitting leaked one rare one-hot category into "
        "`StandardScaler`'s statistics in notebook 1's first draft; this "
        "notebook uses the corrected, leak-free order from the start."
    ),
    code(
        'df_train = pd.read_csv("../../data/train.csv")\n'
        'df_train = df_train[~df_train["Id"].isin([524, 1299])]\n\n'
        "X_raw = df_train[FEATURE_COLS]\n"
        'y = np.log1p(df_train["SalePrice"]).to_numpy()\n\n'
        "X_train_raw, X_val_raw, y_train, y_val = train_test_split(\n"
        "    X_raw, y, test_size=0.2, random_state=42\n"
        ")\n\n"
        "pipeline = build_pipeline()\n"
        "X_train = pipeline.fit_transform(X_train_raw).to_numpy()\n"
        "X_val = pipeline.transform(X_val_raw).to_numpy()\n"
        "X_train.shape, X_val.shape"
    ),
    md(
        "## Alpha sweep\n\n"
        "Each model family's `alpha` is swept independently over the same "
        "log-spaced range -- sklearn's `Ridge`/`Lasso` normalize their "
        "internal objectives differently from each other and from these "
        "from-scratch versions, so a shared numeric `alpha` does not mean "
        "the same shrinkage strength across all four. Comparison happens "
        "at the level of each model's own best validation RMSE, not "
        "matching alpha values 1:1."
    ),
    code(
        "alphas = np.logspace(-3, 1, 9)\n\n\n"
        "def rmse(y_true, y_pred):\n"
        "    return np.sqrt(mean_squared_error(y_true, y_pred))\n\n\n"
        "ridge_gd_rmse = []\n"
        "ridge_sk_rmse = []\n"
        "lasso_gd_rmse = []\n"
        "lasso_sk_rmse = []\n\n"
        "for a in alphas:\n"
        "    ridge_gd = RidgeGD(\n"
        "        learning_rate=0.1, batch_size=32, n_epochs=300, alpha=a,\n"
        "        random_state=42,\n"
        "    )\n"
        "    ridge_gd.fit(X_train, y_train)\n"
        "    ridge_gd_rmse.append(rmse(y_val, ridge_gd.predict(X_val)))\n\n"
        "    ridge_sk = Ridge(alpha=a)\n"
        "    ridge_sk.fit(X_train, y_train)\n"
        "    ridge_sk_rmse.append(rmse(y_val, ridge_sk.predict(X_val)))\n\n"
        "    lasso_gd = LassoGD(\n"
        "        learning_rate=0.1, batch_size=32, n_epochs=300, alpha=a,\n"
        "        random_state=42,\n"
        "    )\n"
        "    lasso_gd.fit(X_train, y_train)\n"
        "    lasso_gd_rmse.append(rmse(y_val, lasso_gd.predict(X_val)))\n\n"
        "    lasso_sk = Lasso(alpha=a)\n"
        "    lasso_sk.fit(X_train, y_train)\n"
        "    lasso_sk_rmse.append(rmse(y_val, lasso_sk.predict(X_val)))\n\n"
        "ridge_gd_rmse = np.array(ridge_gd_rmse)\n"
        "ridge_sk_rmse = np.array(ridge_sk_rmse)\n"
        "lasso_gd_rmse = np.array(lasso_gd_rmse)\n"
        "lasso_sk_rmse = np.array(lasso_sk_rmse)"
    ),
    md(
        "## Validation RMSE vs. alpha\n\n"
        "Where each curve bottoms out is that model's best bias-variance "
        "tradeoff on this holdout split."
    ),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n\n"
        'axes[0].semilogx(alphas, ridge_gd_rmse, marker="o", label="RidgeGD")\n'
        'axes[0].semilogx(alphas, ridge_sk_rmse, marker="o", label="sklearn Ridge")\n'
        'axes[0].set_xlabel("alpha")\n'
        'axes[0].set_ylabel("validation RMSE (log space)")\n'
        'axes[0].set_title("Ridge: validation RMSE vs. alpha")\n'
        "axes[0].legend()\n\n"
        'axes[1].semilogx(alphas, lasso_gd_rmse, marker="o", label="LassoGD")\n'
        'axes[1].semilogx(alphas, lasso_sk_rmse, marker="o", label="sklearn Lasso")\n'
        'axes[1].set_xlabel("alpha")\n'
        'axes[1].set_ylabel("validation RMSE (log space)")\n'
        'axes[1].set_title("Lasso: validation RMSE vs. alpha")\n'
        "axes[1].legend()\n\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "## Refit each model at its own best alpha\n\n"
        "The plain-regression baseline (`LinearRegressionGD` / sklearn "
        "`LinearRegression`) is refit here too, as an unregularized "
        "reference point -- same hyperparameters as "
        "`1_linear_regression.ipynb`."
    ),
    code(
        "best_ridge_gd_alpha = alphas[np.argmin(ridge_gd_rmse)]\n"
        "best_ridge_sk_alpha = alphas[np.argmin(ridge_sk_rmse)]\n"
        "best_lasso_gd_alpha = alphas[np.argmin(lasso_gd_rmse)]\n"
        "best_lasso_sk_alpha = alphas[np.argmin(lasso_sk_rmse)]\n\n"
        "ridge_gd_best = RidgeGD(\n"
        "    learning_rate=0.1, batch_size=32, n_epochs=300,\n"
        "    alpha=best_ridge_gd_alpha, random_state=42,\n"
        ")\n"
        "ridge_gd_best.fit(X_train, y_train)\n\n"
        "ridge_sk_best = Ridge(alpha=best_ridge_sk_alpha)\n"
        "ridge_sk_best.fit(X_train, y_train)\n\n"
        "lasso_gd_best = LassoGD(\n"
        "    learning_rate=0.1, batch_size=32, n_epochs=300,\n"
        "    alpha=best_lasso_gd_alpha, random_state=42,\n"
        ")\n"
        "lasso_gd_best.fit(X_train, y_train)\n\n"
        "lasso_sk_best = Lasso(alpha=best_lasso_sk_alpha)\n"
        "lasso_sk_best.fit(X_train, y_train)\n\n"
        "linreg_gd_baseline = LinearRegressionGD(\n"
        "    learning_rate=0.1, batch_size=32, n_epochs=300, random_state=42\n"
        ")\n"
        "linreg_gd_baseline.fit(X_train, y_train)\n\n"
        "linreg_sk_baseline = LinearRegression()\n"
        "linreg_sk_baseline.fit(X_train, y_train)\n\n"
        "MODELS = [\n"
        '    ("LinearRegressionGD (baseline)", linreg_gd_baseline),\n'
        '    ("sklearn LinearRegression (baseline)", linreg_sk_baseline),\n'
        '    (f"RidgeGD (alpha={best_ridge_gd_alpha:.4g})", ridge_gd_best),\n'
        '    (f"sklearn Ridge (alpha={best_ridge_sk_alpha:.4g})", ridge_sk_best),\n'
        '    (f"LassoGD (alpha={best_lasso_gd_alpha:.4g})", lasso_gd_best),\n'
        '    (f"sklearn Lasso (alpha={best_lasso_sk_alpha:.4g})", lasso_sk_best),\n'
        "]"
    ),
    md(
        "## Metrics table\n\n"
        "RMSE in log space is the metric that actually matters (matches "
        "the Kaggle scoring); RMSE in dollar space (via `expm1`) and MAPE "
        "are for human-readable intuition."
    ),
    code(
        "for name, model in MODELS:\n"
        "    preds_log = model.predict(X_val)\n"
        "    preds_dollars = np.expm1(preds_log)\n"
        "    actual_dollars = np.expm1(y_val)\n"
        "    print(name)\n"
        '    print("  RMSE (log space):", rmse(y_val, preds_log))\n'
        '    print("  RMSE (dollars, intuition only):", rmse(actual_dollars, preds_dollars))\n'
        '    print("  R^2:", r2_score(y_val, preds_log))\n'
        '    print("  MAE (log space):", mean_absolute_error(y_val, preds_log))\n'
        "    mape = mean_absolute_percentage_error(actual_dollars, preds_dollars)\n"
        '    print("  MAPE (dollars, % of actual price):", 100 * mape)'
    ),
    md(
        "## Coefficient sparsity comparison\n\n"
        "Counts how many coefficients each model pushed below a small "
        "magnitude threshold -- expected to show Lasso (both GD and "
        "sklearn) driving meaningfully more coefficients toward zero than "
        "Ridge, directly illustrating feature selection against the "
        "pipeline's 236-column, rank-207 design (`preprocessing_check.ipynb`)."
    ),
    code(
        "threshold = 1e-3\n"
        "sparsity_models = [\n"
        '    ("RidgeGD", ridge_gd_best.coef_),\n'
        '    ("sklearn Ridge", ridge_sk_best.coef_.ravel()),\n'
        '    ("LassoGD", lasso_gd_best.coef_),\n'
        '    ("sklearn Lasso", lasso_sk_best.coef_.ravel()),\n'
        "]\n\n"
        "names = [name for name, _ in sparsity_models]\n"
        "zero_counts = [\n"
        "    int(np.sum(np.abs(coef) < threshold)) for _, coef in sparsity_models\n"
        "]\n\n"
        "plt.bar(names, zero_counts)\n"
        'plt.ylabel(f"# coefficients with |coef| < {threshold}")\n'
        'plt.title("Coefficient sparsity at best alpha")\n'
        "plt.xticks(rotation=20)\n"
        "plt.show()\n\n"
        "n_features = ridge_gd_best.coef_.shape[0]\n"
        "for name, count in zip(names, zero_counts, strict=True):\n"
        '    print(f"{name}: {count} / {n_features} coefficients near zero")'
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
with open("notebooks/models/2_regularized_regression.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
```

- [ ] **Step 2: Run the script, execute the notebook**

Run:
```bash
python3 /path/to/scratchpad/build_regularized_regression_nb.py
cd notebooks/models && uv run --project ../.. jupyter nbconvert --to notebook --execute --inplace 2_regularized_regression.ipynb
```
Expected: `[NbConvertApp] Writing ... bytes to 2_regularized_regression.ipynb`, no error cells.

- [ ] **Step 3: Verify no errors**

Run:
```python
import json

nb = json.load(open("notebooks/models/2_regularized_regression.ipynb"))
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

- [ ] **Step 4: Sanity-check the printed metrics and plots**

Open the executed notebook and confirm: (1) both alpha-sweep curves have a visible minimum somewhere inside the swept range, not monotonically increasing/decreasing across all 9 alphas (which would mean the range needs widening); (2) the 4 regularized models' log-space RMSE are all in the same ballpark as the plain-regression baseline (within a small margin, not orders of magnitude worse — a large gap means a hyperparameter needs adjusting, same check as notebook 1's Step 4); (3) the sparsity bar chart shows LassoGD and sklearn Lasso with visibly more near-zero coefficients than RidgeGD/sklearn Ridge — if Lasso doesn't show *more* sparsity than Ridge at its own best alpha, something about the penalty or alpha range needs revisiting before this task is considered done.

- [ ] **Step 5: Commit**

```bash
git add notebooks/models/2_regularized_regression.ipynb
git commit -m "feat: add notebook comparing RidgeGD/LassoGD against sklearn's Ridge/Lasso"
```

---

## Self-Review Notes

- **Spec coverage:** `RidgeGD` — L2 penalty on `coef_` only, intercept unpenalized, regularized `loss_history_` (Task 1); `LassoGD` — L1 subgradient penalty, same structure, sparsity behavior (Task 2); synthetic-data unit tests for both including Lasso's sparsity assertion (Tasks 1-2); real-data notebook with independent alpha sweeps, best-alpha refit, full metrics table including the plain-regression baseline, and a coefficient-sparsity comparison (Task 3) — every component in the spec has a task.
- **Type consistency checked:** `RidgeGD`/`LassoGD`'s `fit`/`predict` types (`np.ndarray` in, `np.ndarray`/`float` attributes out) match `LinearRegressionGD`'s exactly, and Task 3's notebook calls all three the same way; `build_pipeline()`'s `pd.DataFrame` output is explicitly `.to_numpy()`'d for both the train and validation splits before any model sees it, avoiding the same DataFrame/ndarray mismatch bug class flagged in notebook 1's plan.
- **No placeholders:** every step has real code, real hyperparameters (`alpha=0.1` for Ridge's test, `alpha=0.5` for Lasso's test, `np.logspace(-3, 1, 9)` for the sweep), or a real command.
