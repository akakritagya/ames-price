# ElasticNet Regression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement ElasticNet (combined L1+L2 penalty) regularized linear regression from scratch (mini-batch gradient descent, plain numpy), and compare it against scikit-learn's own `ElasticNet` on the real Ames data, alongside the existing plain/Ridge/Lasso baselines.

**Architecture:** One new estimator class (`ElasticNetGD`), mirroring `RidgeGD`/`LassoGD`'s exact skeleton (zero-init weights, separately tracked intercept, seeded mini-batch loop, `loss_history_`) plus a combined L1+L2 penalty term parameterized by `alpha`/`l1_ratio` (sklearn's convention). A new notebook grid-sweeps `(alpha, l1_ratio)` against validation RMSE, refits at each model's best combo, and reports metrics plus a coefficient-sparsity comparison against the full existing model roster.

**Tech Stack:** Python 3.13.15, numpy, pandas, scikit-learn, matplotlib, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-09-06-elastic-net-regression-design.md`

## Global Constraints

- Python 3.13.15 (`pyproject.toml` `requires-python`); every command below runs via `uv run`.
- ruff: line-length 80, double quotes, rules `E,F,I,UP,B,SIM,RUF,S,PTH,C901,PLR0913,D` (numpy pydocstyle convention) — `uv run ruff format <file>` then `uv run ruff check <file>` must be clean before each commit. Every public class/method needs a numpy-style docstring (`Parameters`/`Returns` sections), matching `src/ames_price/models/ridge_gd.py`'s existing style.
- mypy: `files = ["src"]`, `check_untyped_defs = true` — `uv run mypy src` must be clean; annotate `fit`/`predict` signatures.
- `ElasticNetGD` is a plain class, **not** an `sklearn.base.BaseEstimator`/`TransformerMixin` subclass — same "from scratch" scope as `LinearRegressionGD`/`RidgeGD`/`LassoGD`. Its `fit`/`predict` types are `np.ndarray`, not `pd.DataFrame`.
- Penalizes `coef_` only — the intercept is always updated by the plain unregularized gradient (`(2 / len(batch_idx)) * error.sum()`), exactly as in `LinearRegressionGD`/`RidgeGD`/`LassoGD`.
- Objective: `MSE + alpha * l1_ratio * sum(abs(coef_)) + alpha * 0.5 * (1 - l1_ratio) * sum(coef_ ** 2)`. Gradient penalty term: `alpha * (l1_ratio * np.sign(coef_) + (1 - l1_ratio) * coef_)`, not scaled by batch size. `loss_history_` records this full regularized objective per epoch, not bare MSE.
- At `l1_ratio=1.0` the penalty gradient is bitwise identical to `LassoGD`'s at the same `alpha`. At `l1_ratio=0.0` it is `alpha * coef_`, which is **not** numerically equal to `RidgeGD`'s `2 * alpha * coef_` at the same `alpha` (a 4x difference) — document this explicitly in the docstring as expected, not a bug.
- No proximal/coordinate-descent solver, no cross-validation, no early stopping — fixed `n_epochs`, single train/validation split, exactly as decided in the spec.
- No attempt to make GD's `alpha`/`l1_ratio` numerically match sklearn's own internal scaling — each model's alpha/l1_ratio grid is swept independently; comparison is at the level of best-achievable validation RMSE, not identical hyperparameter meaning.
- No comments explaining *what* code does — only *why*, when non-obvious. Match the rest of the repo.
- Model unit tests use small synthetic numpy arrays with a known linear relationship, not the real CSVs — only the notebook (Task 2) touches real data.

---

## Task 1: `ElasticNetGD` — combined L1+L2 mini-batch gradient descent

**Files:**
- Create: `src/ames_price/models/elastic_net_gd.py`
- Test: `tests/test_elastic_net_gd.py`

**Interfaces:**
- Consumes: nothing from earlier work — pure numpy, structurally identical to `RidgeGD`/`LassoGD` (`src/ames_price/models/ridge_gd.py`, `src/ames_price/models/lasso_gd.py`) except for the penalty term.
- Produces: `class ElasticNetGD` with `__init__(self, learning_rate: float = 0.01, batch_size: int = 32, n_epochs: int = 100, alpha: float = 1.0, l1_ratio: float = 0.5, random_state: int | None = None)`, `fit(self, X: np.ndarray, y: np.ndarray) -> ElasticNetGD` (sets `coef_: np.ndarray`, `intercept_: float`, `loss_history_: list[float]`), `predict(self, X: np.ndarray) -> np.ndarray`. Task 2's notebook imports and calls this exactly as it calls `RidgeGD`/`LassoGD`/`LinearRegressionGD` and sklearn's own `ElasticNet`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_elastic_net_gd.py
import numpy as np

from ames_price.models.elastic_net_gd import ElasticNetGD


def _sparse_linear_fixture(
    n: int = 200, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    # x1, x2 are informative; x3, x4 have a true coefficient of exactly
    # zero -- same fixture as test_lasso_gd.py, reused here since it
    # exercises both recovery and sparsity in one dataset.
    true_coef = np.array([3.0, -2.0, 0.0, 0.0])
    true_intercept = 5.0
    noise = rng.normal(scale=0.01, size=n)
    y = X @ true_coef + true_intercept + noise
    return X, y, true_coef, true_intercept


def test_l1_ratio_one_matches_lasso_behavior():
    X, y, true_coef, true_intercept = _sparse_linear_fixture()
    model = ElasticNetGD(
        learning_rate=0.1,
        batch_size=20,
        n_epochs=300,
        alpha=0.5,
        l1_ratio=1.0,
        random_state=42,
    )
    model.fit(X, y)

    # at l1_ratio=1.0 the penalty gradient is bitwise identical to
    # LassoGD's -- same qualitative behavior as test_lasso_gd.py's
    # sparsity test at the same alpha.
    assert abs(model.coef_[2]) < 0.1
    assert abs(model.coef_[3]) < 0.1
    assert np.allclose(model.coef_[:2], true_coef[:2], atol=0.5)
    assert abs(model.intercept_ - true_intercept) < 0.1


def test_l1_ratio_half_shows_mixed_shrinkage():
    X, y, true_coef, true_intercept = _sparse_linear_fixture()
    model = ElasticNetGD(
        learning_rate=0.1,
        batch_size=20,
        n_epochs=300,
        alpha=0.5,
        l1_ratio=0.5,
        random_state=42,
    )
    model.fit(X, y)

    # the added L2 term biases every coefficient toward zero, not just
    # the true-zero ones -- looser tolerance than the pure-L1 case.
    assert np.allclose(model.coef_[:2], true_coef[:2], atol=0.8)
    # x3/x4 still shrink toward zero, though less tightly than l1_ratio=1.
    assert abs(model.coef_[2]) < 0.3
    assert abs(model.coef_[3]) < 0.3
    assert abs(model.intercept_ - true_intercept) < 0.1


def test_predict_output_shape_matches_input_rows():
    X, y, _, _ = _sparse_linear_fixture()
    model = ElasticNetGD(n_epochs=10, random_state=42)
    model.fit(X, y)

    X_new = np.random.default_rng(0).normal(size=(5, 4))
    predictions = model.predict(X_new)

    assert predictions.shape == (5,)


def test_loss_history_length_and_non_increasing():
    X, y, _, _ = _sparse_linear_fixture()
    model = ElasticNetGD(
        learning_rate=0.01,
        batch_size=len(X),
        n_epochs=200,
        alpha=0.5,
        l1_ratio=0.5,
        random_state=42,
    )
    model.fit(X, y)

    assert len(model.loss_history_) == 200
    assert all(
        model.loss_history_[i + 1] <= model.loss_history_[i] + 1e-9
        for i in range(len(model.loss_history_) - 1)
    )
```

`test_loss_history_length_and_non_increasing` uses the same smaller
`learning_rate` (0.01) as `test_lasso_gd.py`'s equivalent test, for the
same reason: the L1 component's constant-magnitude subgradient can
otherwise cause a small oscillation right around convergence; full-batch
GD at this smaller step size on this well-conditioned problem stays
monotonically non-increasing.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_elastic_net_gd.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ames_price.models.elastic_net_gd'`

- [ ] **Step 3: Write the implementation**

```python
# src/ames_price/models/elastic_net_gd.py
"""ElasticNet-regularized linear regression fit by mini-batch gradient descent."""

from __future__ import annotations

import numpy as np


class ElasticNetGD:
    """Linear regression with a combined L1+L2 penalty, fit by mini-batch gradient descent.

    Same skeleton as ``LinearRegressionGD``/``RidgeGD``/``LassoGD``:
    weights start at zero, the intercept is tracked as its own scalar,
    and the batch loop is identical. The penalty is
    ``alpha * (l1_ratio * sum(abs(coef_)) + 0.5 * (1 - l1_ratio) * sum(coef_ ** 2))``,
    matching scikit-learn's ``ElasticNet`` parameterization -- ``alpha``
    is the overall penalty strength, ``l1_ratio`` in ``[0, 1]`` is the
    mixing weight between the L1 and L2 terms. Excluded from the
    intercept's update, same reasoning as ``RidgeGD``/``LassoGD``.

    At ``l1_ratio=1.0`` the penalty gradient (``alpha * np.sign(coef_)``)
    is bitwise identical to ``LassoGD``'s at the same ``alpha``. At
    ``l1_ratio=0.0`` it is ``alpha * coef_``, which is **not**
    numerically equal to ``RidgeGD``'s ``2 * alpha * coef_`` at the same
    ``alpha`` (a 4x difference in effective L2 strength) -- this is
    expected, not a bug: alpha is not comparable 1:1 across model
    families, consistent with ``RidgeGD``/``LassoGD``'s own docstrings.

    Like ``LassoGD``, the L1 component uses the subgradient approximation
    (``np.sign(0) == 0``) rather than a proximal/soft-thresholding
    solver, so sparsity is soft: coefficients shrink toward zero and
    settle very close to it, but "near zero" (checked with a tolerance),
    not bitwise ``0.0``, is the correct thing to assert or expect.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        n_epochs: int = 100,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
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
            Overall penalty strength applied to coef_. The intercept is
            never penalized.
        l1_ratio : float, default=0.5
            Mixing weight between the L1 and L2 penalty terms, in
            [0, 1]. 1.0 is pure L1 (matches LassoGD's gradient exactly),
            0.0 is pure L2 (shaped like RidgeGD's but not numerically
            equal to it -- see class docstring).
        random_state : int or None, default=None
            Seed for the per-epoch row shuffle. None gives a different,
            non-reproducible shuffle on every fit() call.
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> ElasticNetGD:
        """Fit by mini-batch gradient descent on the ElasticNet-penalized objective.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        ElasticNetGD
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
                ) + self.alpha * (
                    self.l1_ratio * np.sign(self.coef_)
                    + (1 - self.l1_ratio) * self.coef_
                )
                grad_intercept = (2 / len(batch_idx)) * error.sum()

                self.coef_ -= self.learning_rate * grad_coef
                self.intercept_ -= self.learning_rate * grad_intercept

            epoch_pred = X @ self.coef_ + self.intercept_
            epoch_mse = float(np.mean((epoch_pred - y) ** 2))
            epoch_l1 = float(np.sum(np.abs(self.coef_)))
            epoch_l2 = float(np.sum(self.coef_**2))
            epoch_penalty = self.alpha * (
                self.l1_ratio * epoch_l1 + 0.5 * (1 - self.l1_ratio) * epoch_l2
            )
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

Run: `uv run pytest tests/test_elastic_net_gd.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Lint, format, typecheck**

Run: `uv run ruff format src/ames_price/models/elastic_net_gd.py tests/test_elastic_net_gd.py && uv run ruff check src/ames_price/models/elastic_net_gd.py tests/test_elastic_net_gd.py && uv run mypy src`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/ames_price/models/elastic_net_gd.py tests/test_elastic_net_gd.py
git commit -m "feat: add ElasticNetGD (L1+L2-regularized mini-batch gradient descent)"
```

---

## Task 2: `notebooks/models/3_elastic_net.ipynb`

**Files:**
- Create: `notebooks/models/3_elastic_net.ipynb`
- Create (temporary, scratchpad): a small Python script that builds the notebook JSON, matching the pattern used for `notebooks/models/2_regularized_regression.ipynb`

**Interfaces:**
- Consumes: `build_pipeline()` from `ames_price.pipeline`; `LinearRegressionGD` from `ames_price.models.linear_regression`; `RidgeGD` from `ames_price.models.ridge_gd`; `LassoGD` from `ames_price.models.lasso_gd`; `ElasticNetGD` from `ames_price.models.elastic_net_gd` (Task 1); `NUMERIC_COLS`/`ORDINAL_COLS`/`NOMINAL_COLS` from `ames_price.constants`; sklearn's `ElasticNet`, `Ridge`, `Lasso`, `LinearRegression`.
- Produces: an executed, committed notebook — no code elsewhere depends on it.

- [ ] **Step 1: Write the notebook-build script**

```python
# scratchpad script (not committed) -- builds
# notebooks/models/3_elastic_net.ipynb
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
        "# ElasticNet: from scratch vs. scikit-learn\n\n"
        "Fits `ElasticNetGD` (mini-batch gradient descent, plain numpy) "
        "and scikit-learn's own `ElasticNet` on the same preprocessed "
        "data and target, alongside the plain-regression/Ridge/Lasso "
        "baselines from `1_linear_regression.ipynb`/"
        "`2_regularized_regression.ipynb` -- per "
        "`docs/superpowers/specs/2026-09-06-elastic-net-regression-design.md`."
    ),
    code(
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n\n"
        "from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge\n"
        "from sklearn.metrics import (\n"
        "    mean_absolute_error,\n"
        "    mean_absolute_percentage_error,\n"
        "    mean_squared_error,\n"
        "    r2_score,\n"
        ")\n"
        "from sklearn.model_selection import train_test_split\n\n"
        "from ames_price.constants import NOMINAL_COLS, NUMERIC_COLS, ORDINAL_COLS\n"
        "from ames_price.models.elastic_net_gd import ElasticNetGD\n"
        "from ames_price.models.lasso_gd import LassoGD\n"
        "from ames_price.models.linear_regression import LinearRegressionGD\n"
        "from ames_price.models.ridge_gd import RidgeGD\n"
        "from ames_price.pipeline import build_pipeline\n\n"
        "FEATURE_COLS = NUMERIC_COLS + ORDINAL_COLS + NOMINAL_COLS"
    ),
    md(
        "## Load data, drop outliers, build the target, split before "
        "fitting\n\n"
        "Same leak-free setup as notebooks 1 and 2: `Id` 524/1299 "
        "dropped, target is `log_sale_price`, pipeline `fit_transform` "
        "on the training split only, `transform` on validation."
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
        "## Grid sweep over (alpha, l1_ratio)\n\n"
        "A 2D grid, narrower than Ridge/Lasso's 9-point 1D sweep since "
        "there are now two hyperparameters (5 alphas x 5 l1_ratios = 25 "
        "combinations). As with Ridge/Lasso, `ElasticNetGD`'s and "
        "sklearn `ElasticNet`'s alpha/l1_ratio are not expected to mean "
        "exactly the same shrinkage strength -- comparison happens at "
        "the level of each model's own best validation RMSE."
    ),
    code(
        "alphas = np.logspace(-3, 1, 5)\n"
        "l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]\n\n\n"
        "def rmse(y_true, y_pred):\n"
        "    return np.sqrt(mean_squared_error(y_true, y_pred))\n\n\n"
        "en_gd_rmse = np.zeros((len(alphas), len(l1_ratios)))\n"
        "en_sk_rmse = np.zeros((len(alphas), len(l1_ratios)))\n\n"
        "for i, a in enumerate(alphas):\n"
        "    for j, r in enumerate(l1_ratios):\n"
        "        en_gd = ElasticNetGD(\n"
        "            learning_rate=0.1, batch_size=32, n_epochs=300,\n"
        "            alpha=a, l1_ratio=r, random_state=42,\n"
        "        )\n"
        "        en_gd.fit(X_train, y_train)\n"
        "        en_gd_rmse[i, j] = rmse(y_val, en_gd.predict(X_val))\n\n"
        "        en_sk = ElasticNet(alpha=a, l1_ratio=r)\n"
        "        en_sk.fit(X_train, y_train)\n"
        "        en_sk_rmse[i, j] = rmse(y_val, en_sk.predict(X_val))"
    ),
    md(
        "## Validation RMSE heatmap\n\n"
        "The 2D analogue of notebook 2's `semilogx` line plot -- darkest "
        "cell is each model's best (alpha, l1_ratio) region on this "
        "holdout split."
    ),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n\n"
        'for ax, data, title in [\n'
        '    (axes[0], en_gd_rmse, "ElasticNetGD"),\n'
        '    (axes[1], en_sk_rmse, "sklearn ElasticNet"),\n'
        "]:\n"
        '    im = ax.imshow(data, aspect="auto", cmap="viridis_r")\n'
        "    ax.set_xticks(range(len(l1_ratios)))\n"
        "    ax.set_xticklabels(l1_ratios)\n"
        "    ax.set_yticks(range(len(alphas)))\n"
        '    ax.set_yticklabels([f"{a:.4g}" for a in alphas])\n'
        '    ax.set_xlabel("l1_ratio")\n'
        '    ax.set_ylabel("alpha")\n'
        "    ax.set_title(f\"{title}: validation RMSE\")\n"
        "    fig.colorbar(im, ax=ax)\n\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "## Refit each model at its own best (alpha, l1_ratio)\n\n"
        "The plain/Ridge/Lasso baselines are refit here too, exactly as "
        "in `2_regularized_regression.ipynb`, so this notebook stands on "
        "its own with the full model roster."
    ),
    code(
        "best_en_gd_idx = np.unravel_index(np.argmin(en_gd_rmse), en_gd_rmse.shape)\n"
        "best_en_sk_idx = np.unravel_index(np.argmin(en_sk_rmse), en_sk_rmse.shape)\n"
        "best_en_gd_alpha, best_en_gd_l1_ratio = (\n"
        "    alphas[best_en_gd_idx[0]], l1_ratios[best_en_gd_idx[1]],\n"
        ")\n"
        "best_en_sk_alpha, best_en_sk_l1_ratio = (\n"
        "    alphas[best_en_sk_idx[0]], l1_ratios[best_en_sk_idx[1]],\n"
        ")\n\n"
        "en_gd_best = ElasticNetGD(\n"
        "    learning_rate=0.1, batch_size=32, n_epochs=300,\n"
        "    alpha=best_en_gd_alpha, l1_ratio=best_en_gd_l1_ratio,\n"
        "    random_state=42,\n"
        ")\n"
        "en_gd_best.fit(X_train, y_train)\n\n"
        "en_sk_best = ElasticNet(\n"
        "    alpha=best_en_sk_alpha, l1_ratio=best_en_sk_l1_ratio,\n"
        ")\n"
        "en_sk_best.fit(X_train, y_train)\n\n"
        "# best alphas below are copied from 2_regularized_regression.ipynb's\n"
        "# own executed output (its alpha sweep), not re-swept here.\n"
        "ridge_gd_best = RidgeGD(\n"
        "    learning_rate=0.1, batch_size=32, n_epochs=300, alpha=0.03162,\n"
        "    random_state=42,\n"
        ")\n"
        "ridge_gd_best.fit(X_train, y_train)\n\n"
        "ridge_sk_best = Ridge(alpha=31.62)\n"
        "ridge_sk_best.fit(X_train, y_train)\n\n"
        "lasso_gd_best = LassoGD(\n"
        "    learning_rate=0.1, batch_size=32, n_epochs=300, alpha=0.01,\n"
        "    random_state=42,\n"
        ")\n"
        "lasso_gd_best.fit(X_train, y_train)\n\n"
        "lasso_sk_best = Lasso(alpha=0.003162)\n"
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
        '    ("RidgeGD (alpha=0.03162)", ridge_gd_best),\n'
        '    ("sklearn Ridge (alpha=31.62)", ridge_sk_best),\n'
        '    ("LassoGD (alpha=0.01)", lasso_gd_best),\n'
        '    ("sklearn Lasso (alpha=0.003162)", lasso_sk_best),\n'
        '    (\n'
        '        f"ElasticNetGD (alpha={best_en_gd_alpha:.4g}, "\n'
        '        f"l1_ratio={best_en_gd_l1_ratio})",\n'
        "        en_gd_best,\n"
        "    ),\n"
        '    (\n'
        '        f"sklearn ElasticNet (alpha={best_en_sk_alpha:.4g}, "\n'
        '        f"l1_ratio={best_en_sk_l1_ratio})",\n'
        "        en_sk_best,\n"
        "    ),\n"
        "]"
    ),
    md(
        "## Metrics table\n\n"
        "RMSE in log space is the metric that actually matters (matches "
        "the Kaggle scoring); RMSE in dollar space (via `expm1`) and "
        "MAPE are for human-readable intuition."
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
        "Extends `2_regularized_regression.ipynb`'s bar chart with "
        "ElasticNet -- expected to sit between Ridge (least sparse) and "
        "Lasso (most sparse), illustrating the mixed penalty's "
        "intermediate feature-selection behavior against the pipeline's "
        "236-column, rank-207 design (`preprocessing_check.ipynb`)."
    ),
    code(
        "threshold = 1e-3\n"
        "sparsity_models = [\n"
        '    ("RidgeGD", ridge_gd_best.coef_),\n'
        '    ("sklearn Ridge", ridge_sk_best.coef_.ravel()),\n'
        '    ("ElasticNetGD", en_gd_best.coef_),\n'
        '    ("sklearn ElasticNet", en_sk_best.coef_.ravel()),\n'
        '    ("LassoGD", lasso_gd_best.coef_),\n'
        '    ("sklearn Lasso", lasso_sk_best.coef_.ravel()),\n'
        "]\n\n"
        "names = [name for name, _ in sparsity_models]\n"
        "zero_counts = [\n"
        "    int(np.sum(np.abs(coef) < threshold)) for _, coef in sparsity_models\n"
        "]\n\n"
        "plt.bar(names, zero_counts)\n"
        'plt.ylabel(f"# coefficients with |coef| < {threshold}")\n'
        'plt.title("Coefficient sparsity at best hyperparameters")\n'
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
with open("notebooks/models/3_elastic_net.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
```

The Ridge/Lasso refit alphas hardcoded above
(`ridge_gd_best`/`ridge_sk_best` at 0.03162/31.62,
`lasso_gd_best`/`lasso_sk_best` at 0.01/0.003162) are copied directly
from `2_regularized_regression.ipynb`'s own executed output (its
`for name, model in MODELS: print(name)` cell) — not re-swept here,
since that sweep already happened and committed real results.

- [ ] **Step 2: Run the script, execute the notebook**

Run:
```bash
python3 /path/to/scratchpad/build_elastic_net_nb.py
cd notebooks/models && uv run --project ../.. jupyter nbconvert --to notebook --execute --inplace 3_elastic_net.ipynb
```
Expected: `[NbConvertApp] Writing ... bytes to 3_elastic_net.ipynb`, no error cells.

- [ ] **Step 3: Verify no errors**

Run:
```python
import json

nb = json.load(open("notebooks/models/3_elastic_net.ipynb"))
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

Open the executed notebook and confirm: (1) both heatmaps show a
visible minimum somewhere inside the swept 5x5 grid, not at a corner
(a corner minimum means the range needs widening in either dimension);
(2) `ElasticNetGD`/sklearn `ElasticNet`'s log-space RMSE are in the same
ballpark as the other regularized models (within a small margin, not
orders of magnitude worse); (3) the sparsity bar chart shows
`ElasticNetGD`/sklearn `ElasticNet` sitting between Ridge (fewer
near-zero coefficients) and Lasso (more near-zero coefficients) — if
ElasticNet doesn't land between the two, the l1_ratio grid or alpha
range needs revisiting before this task is considered done.

- [ ] **Step 5: Commit**

```bash
git add notebooks/models/3_elastic_net.ipynb
git commit -m "feat: add notebook comparing ElasticNetGD against sklearn's ElasticNet"
```

---

## Self-Review Notes

- **Spec coverage:** `ElasticNetGD` — combined L1+L2 penalty on `coef_` only, intercept unpenalized, regularized `loss_history_`, `l1_ratio=1.0` matching `LassoGD` exactly, `l1_ratio=0.0` documented as *not* matching `RidgeGD` numerically (Task 1); synthetic-data unit tests covering the pure-L1 edge case and the mixed case (Task 1); real-data notebook with an independent 2D grid sweep, best-combo refit, full metrics table including the plain/Ridge/Lasso baselines, and a coefficient-sparsity comparison expected to land between Ridge and Lasso (Task 2) — every component in the spec has a task.
- **Type consistency checked:** `ElasticNetGD`'s `fit`/`predict` types (`np.ndarray` in, `np.ndarray`/`float` attributes out) match `RidgeGD`/`LassoGD`/`LinearRegressionGD`'s exactly, and Task 2's notebook calls it the same way as the other three model families; `build_pipeline()`'s `pd.DataFrame` output is explicitly `.to_numpy()`'d for both splits before any model sees it, matching notebook 2's leak-free order.
- **No placeholders:** every step has real code and real hyperparameters (`alpha=0.5`/`l1_ratio=1.0` and `alpha=0.5`/`l1_ratio=0.5` for Task 1's tests, `np.logspace(-3, 1, 5)` x `[0.1, 0.3, 0.5, 0.7, 0.9]` for the grid sweep, and Task 2's Ridge/Lasso refit alphas — `0.03162`/`31.62`/`0.01`/`0.003162` — copied directly from `2_regularized_regression.ipynb`'s own executed output rather than left as placeholders).
