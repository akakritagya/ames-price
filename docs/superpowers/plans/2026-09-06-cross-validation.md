# Cross-Validated Hyperparameter Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace notebooks 2/3's single train/validation-split alpha (and alpha/l1_ratio) sweeps with 5-fold cross-validation for every regularized model (`RidgeGD`, sklearn `Ridge`, `LassoGD`, sklearn `Lasso`, `ElasticNetGD`, sklearn `ElasticNet`), in a new notebook that also compares the CV-selected hyperparameter against notebooks 2/3's single-split choice.

**Architecture:** One new from-scratch utility, `kfold_split()`, mirroring the GD models' own shuffle style (`np.random.default_rng`). A new notebook uses it inside a reusable `cv_scores()` helper (fresh `build_pipeline()` per fold, to stay leak-free) to score every alpha/l1_ratio candidate across 5 folds, picks each model's best mean-CV-RMSE combo, refits once on the full CV pool, and reports final-holdout metrics plus a CV-vs-single-split comparison table.

**Tech Stack:** Python 3.13.15, numpy, pandas, scikit-learn, matplotlib, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-09-06-cross-validation-design.md`

## Global Constraints

- Python 3.13.15 (`pyproject.toml` `requires-python`); every command below runs via `uv run`.
- ruff: line-length 87, double quotes, rules `E,F,I,UP,B,SIM,RUF,S,PTH,C901,PLR0913,D` (numpy pydocstyle convention) — `uv run ruff format <file>` then `uv run ruff check <file>` must be clean before each commit. Every public function needs a numpy-style docstring (`Parameters`/`Returns` sections).
- mypy: `files = ["src"]`, `check_untyped_defs = true` — `uv run mypy src` must be clean; annotate `kfold_split`'s signature.
- `kfold_split` is index-only — it never touches `X`/`y`, only `range(n_samples)`. Same reasoning as this project's existing use of `train_test_split`: split first (as indices/rows), transform per-split after.
- `kfold_split` uses the same shuffle style as the GD models: `np.random.default_rng(random_state)` then `rng.permutation(n_samples)`, so a fixed `random_state` reproduces the same folds.
- No new alpha/l1_ratio grids: reuse notebook 2's `alphas_gd = np.logspace(-3, 1, 9)` (`RidgeGD`/`LassoGD`) and `alphas_sk = np.logspace(-3, 3, 9)` (sklearn `Ridge` only — sklearn `Lasso` uses `alphas_gd`, matching notebook 2's own choice), and notebook 3's `alphas_en = np.logspace(-3, 1, 5)` x `l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]` for ElasticNet.
- The pipeline is refit fresh inside every fold (`build_pipeline().fit_transform()` on that fold's training rows, `.transform()` on its held-out rows) — never reuse one global pipeline fit across folds. This is the leak-free-fitting principle already established in notebooks 1-3, extended to CV.
- The CV sweep loops must NOT pass `verbose=True` to any GD model (would print across hundreds of fold fits). Only the final single refit-at-best-alpha step uses `verbose=True`, matching notebooks 2/3's existing convention.
- `LinearRegressionGD`/sklearn `LinearRegression` are refit once on the full CV pool as an unregularized reference point but are **not** cross-validated (no penalty hyperparameter to select) — same treatment as notebooks 2/3.
- The single-split alpha/l1_ratio values used for the CV-vs-single-split comparison table are hardcoded, copied directly from notebook 2's and notebook 3's own executed output — not re-derived: `RidgeGD` 0.03162, sklearn `Ridge` 31.62, `LassoGD` 0.01, sklearn `Lasso` 0.003162, `ElasticNetGD` alpha=0.01/l1_ratio=0.7, sklearn `ElasticNet` alpha=0.01/l1_ratio=0.3.
- Metrics tables use the `pd.DataFrame`-per-row pattern already established in notebooks 1-3 (a `metrics_rows` list of dicts, `pd.DataFrame(metrics_rows).set_index("model")`), not `print()` loops.
- No comments explaining *what* code does — only *why*, when non-obvious. Match the rest of the repo.
- `kfold_split` unit tests use small synthetic sample counts (e.g. 20, 23), not the real CSV — only the notebook (Task 2) touches real data.

---

## Task 1: `kfold_split` — from-scratch k-fold index splitting

**Files:**
- Create: `src/ames_price/model_selection.py`
- Test: `tests/test_model_selection.py`

**Interfaces:**
- Consumes: nothing from earlier work — pure numpy.
- Produces: `def kfold_split(n_samples: int, n_splits: int = 5, random_state: int | None = None) -> list[tuple[np.ndarray, np.ndarray]]`. Task 2's notebook imports this and calls it as `kfold_split(len(X_raw), n_splits=5, random_state=42)` inside its `cv_scores()` helper.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_model_selection.py
import numpy as np

from ames_price.model_selection import kfold_split


def test_folds_partition_all_indices_exactly_once():
    folds = kfold_split(n_samples=23, n_splits=5, random_state=42)

    all_val_idx = np.concatenate([val_idx for _, val_idx in folds])
    assert sorted(all_val_idx.tolist()) == list(range(23))


def test_train_and_val_disjoint_within_each_fold():
    folds = kfold_split(n_samples=23, n_splits=5, random_state=42)

    for train_idx, val_idx in folds:
        assert set(train_idx.tolist()).isdisjoint(set(val_idx.tolist()))
        assert len(train_idx) + len(val_idx) == 23


def test_fold_sizes_as_equal_as_possible():
    folds = kfold_split(n_samples=23, n_splits=5, random_state=42)

    # 23 = 5*4 + 3 -- three folds get 5 rows, two folds get 4.
    val_sizes = sorted(len(val_idx) for _, val_idx in folds)
    assert val_sizes == [4, 4, 5, 5, 5]


def test_fold_sizes_all_equal_when_evenly_divisible():
    folds = kfold_split(n_samples=20, n_splits=5, random_state=42)

    assert [len(val_idx) for _, val_idx in folds] == [4, 4, 4, 4, 4]


def test_same_random_state_reproduces_identical_folds():
    folds_a = kfold_split(n_samples=23, n_splits=5, random_state=7)
    folds_b = kfold_split(n_samples=23, n_splits=5, random_state=7)

    for (train_a, val_a), (train_b, val_b) in zip(
        folds_a, folds_b, strict=True
    ):
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(val_a, val_b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_model_selection.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ames_price.model_selection'`

- [ ] **Step 3: Write the implementation**

```python
# src/ames_price/model_selection.py
"""From-scratch k-fold index splitting for cross-validation."""

from __future__ import annotations

import numpy as np


def kfold_split(
    n_samples: int, n_splits: int = 5, random_state: int | None = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Shuffle 0..n_samples-1, then split into n_splits folds.

    Index-only: never touches X/y directly. Apply the returned indices
    to your own arrays/DataFrames (e.g. ``X_raw.iloc[train_idx]``).

    Parameters
    ----------
    n_samples : int
        Number of rows to split, over the index range range(n_samples).
    n_splits : int, default=5
        Number of folds.
    random_state : int or None, default=None
        Seed for the shuffle. None gives a different, non-reproducible
        shuffle on every call.

    Returns
    -------
    list of (train_idx, val_idx)
        One pair per fold, val_idx is that fold's held-out indices,
        train_idx is every other index. Folds partition
        range(n_samples) exactly once; sizes differ by at most one row
        when n_samples is not evenly divisible by n_splits (the first
        n_samples % n_splits folds get one extra row).
    """
    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(n_samples)

    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1

    folds = []
    start = 0
    for size in fold_sizes:
        val_idx = shuffled[start : start + size]
        train_idx = np.concatenate(
            [shuffled[:start], shuffled[start + size :]]
        )
        folds.append((train_idx, val_idx))
        start += size

    return folds
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_model_selection.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Lint, format, typecheck**

Run: `uv run ruff format src/ames_price/model_selection.py tests/test_model_selection.py && uv run ruff check src/ames_price/model_selection.py tests/test_model_selection.py && uv run mypy src`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/ames_price/model_selection.py tests/test_model_selection.py
git commit -m "feat: add kfold_split for cross-validation"
```

---

## Task 2: `notebooks/models/4_cross_validation.ipynb`

**Files:**
- Create: `notebooks/models/4_cross_validation.ipynb`
- Create (temporary, scratchpad): a small Python script that builds the notebook JSON, matching the pattern used for notebooks 2 and 3

**Interfaces:**
- Consumes: `kfold_split` from `ames_price.model_selection` (Task 1); `build_pipeline()` from `ames_price.pipeline`; `LinearRegressionGD`, `RidgeGD`, `LassoGD`, `ElasticNetGD` from `ames_price.models.*`; `NUMERIC_COLS`/`ORDINAL_COLS`/`NOMINAL_COLS` from `ames_price.constants`; sklearn's `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`.
- Produces: an executed, committed notebook — no code elsewhere depends on it.

- [ ] **Step 1: Write the notebook-build script**

```python
# scratchpad script (not committed) -- builds
# notebooks/models/4_cross_validation.ipynb
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
        "# Cross-validated hyperparameter selection\n\n"
        "Replaces `2_regularized_regression.ipynb`/`3_elastic_net.ipynb`'s "
        "single train/validation-split alpha (and alpha/l1_ratio) sweeps "
        "with 5-fold cross-validation, for every regularized model "
        "(`RidgeGD`/sklearn `Ridge`, `LassoGD`/sklearn `Lasso`, "
        "`ElasticNetGD`/sklearn `ElasticNet`) -- per "
        "`docs/superpowers/specs/2026-09-06-cross-validation-design.md`."
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
        "from ames_price.model_selection import kfold_split\n"
        "from ames_price.models.elastic_net_gd import ElasticNetGD\n"
        "from ames_price.models.lasso_gd import LassoGD\n"
        "from ames_price.models.linear_regression import LinearRegressionGD\n"
        "from ames_price.models.ridge_gd import RidgeGD\n"
        "from ames_price.pipeline import build_pipeline\n\n"
        "FEATURE_COLS = NUMERIC_COLS + ORDINAL_COLS + NOMINAL_COLS"
    ),
    md(
        "## Load data, drop outliers, build the target, split CV pool "
        "vs. final holdout\n\n"
        "`Id` 524/1299 dropped, target is `log_sale_price`, same as "
        "notebooks 1-3. The 80/20 split below uses the same "
        "`random_state=42` as notebooks 2/3's own train/validation "
        "split, so this notebook's final holdout is the exact same rows "
        "-- CV only ever touches the 80% `cv_pool`; the holdout stays "
        "untouched until the final metrics table."
    ),
    code(
        'df_train = pd.read_csv("../../data/train.csv")\n'
        'df_train = df_train[~df_train["Id"].isin([524, 1299])]\n\n'
        "X_raw = df_train[FEATURE_COLS]\n"
        'y = np.log1p(df_train["SalePrice"]).to_numpy()\n\n'
        "cv_pool_raw, holdout_raw, y_cv_pool, y_holdout = train_test_split(\n"
        "    X_raw, y, test_size=0.2, random_state=42\n"
        ")\n"
        "cv_pool_raw.shape, holdout_raw.shape"
    ),
    md(
        "## Cross-validation helper\n\n"
        "`cv_scores` fits a fresh `build_pipeline()` inside every fold -- "
        "`StandardScaler`'s mean/std must never see that fold's held-out "
        "rows, even indirectly through statistics computed over the "
        "whole `cv_pool`. `model_cls(**model_kwargs)` is likewise "
        "constructed fresh per fold, so no fitted state leaks between "
        "folds. Returns one RMSE (log space) per fold; every sweep below "
        "reduces this to a mean and std."
    ),
    code(
        "def rmse(y_true, y_pred):\n"
        "    return np.sqrt(mean_squared_error(y_true, y_pred))\n\n\n"
        "def cv_scores(model_cls, model_kwargs, X_raw, y, n_splits=5, random_state=42):\n"
        "    fold_rmses = []\n"
        "    folds = kfold_split(len(X_raw), n_splits, random_state)\n"
        "    for train_idx, val_idx in folds:\n"
        "        pipeline = build_pipeline()\n"
        "        X_fold_train = pipeline.fit_transform(\n"
        "            X_raw.iloc[train_idx]\n"
        "        ).to_numpy()\n"
        "        X_fold_val = pipeline.transform(X_raw.iloc[val_idx]).to_numpy()\n"
        "        y_fold_train, y_fold_val = y[train_idx], y[val_idx]\n\n"
        "        model = model_cls(**model_kwargs)\n"
        "        model.fit(X_fold_train, y_fold_train)\n"
        "        fold_rmses.append(rmse(y_fold_val, model.predict(X_fold_val)))\n\n"
        "    return np.array(fold_rmses)"
    ),
    md(
        "## Ridge/Lasso: 5-fold CV over alpha\n\n"
        "Same `alphas_gd`/`alphas_sk` ranges as "
        "`2_regularized_regression.ipynb` -- sklearn `Ridge`'s own "
        "minimum sits near alpha=30 there, past `alphas_gd`'s range, so "
        "it alone sweeps the wider `alphas_sk`; `RidgeGD`, `LassoGD`, "
        "and sklearn `Lasso` all sweep `alphas_gd`."
    ),
    code(
        "alphas_gd = np.logspace(-3, 1, 9)\n"
        "alphas_sk = np.logspace(-3, 3, 9)\n\n"
        "ridge_gd_cv_mean, ridge_gd_cv_std = [], []\n"
        "lasso_gd_cv_mean, lasso_gd_cv_std = [], []\n"
        "lasso_sk_cv_mean, lasso_sk_cv_std = [], []\n\n"
        "for a in alphas_gd:\n"
        "    scores = cv_scores(\n"
        "        RidgeGD,\n"
        "        dict(\n"
        "            learning_rate=0.01, batch_size=32, n_epochs=300,\n"
        "            alpha=a, random_state=42,\n"
        "        ),\n"
        "        cv_pool_raw, y_cv_pool,\n"
        "    )\n"
        "    ridge_gd_cv_mean.append(scores.mean())\n"
        "    ridge_gd_cv_std.append(scores.std())\n\n"
        "    scores = cv_scores(\n"
        "        LassoGD,\n"
        "        dict(\n"
        "            learning_rate=0.01, batch_size=32, n_epochs=300,\n"
        "            alpha=a, random_state=42,\n"
        "        ),\n"
        "        cv_pool_raw, y_cv_pool,\n"
        "    )\n"
        "    lasso_gd_cv_mean.append(scores.mean())\n"
        "    lasso_gd_cv_std.append(scores.std())\n\n"
        "    scores = cv_scores(Lasso, dict(alpha=a), cv_pool_raw, y_cv_pool)\n"
        "    lasso_sk_cv_mean.append(scores.mean())\n"
        "    lasso_sk_cv_std.append(scores.std())\n\n"
        "ridge_sk_cv_mean, ridge_sk_cv_std = [], []\n"
        "for a in alphas_sk:\n"
        "    scores = cv_scores(Ridge, dict(alpha=a), cv_pool_raw, y_cv_pool)\n"
        "    ridge_sk_cv_mean.append(scores.mean())\n"
        "    ridge_sk_cv_std.append(scores.std())\n\n"
        "ridge_gd_cv_mean = np.array(ridge_gd_cv_mean)\n"
        "ridge_gd_cv_std = np.array(ridge_gd_cv_std)\n"
        "ridge_sk_cv_mean = np.array(ridge_sk_cv_mean)\n"
        "ridge_sk_cv_std = np.array(ridge_sk_cv_std)\n"
        "lasso_gd_cv_mean = np.array(lasso_gd_cv_mean)\n"
        "lasso_gd_cv_std = np.array(lasso_gd_cv_std)\n"
        "lasso_sk_cv_mean = np.array(lasso_sk_cv_mean)\n"
        "lasso_sk_cv_std = np.array(lasso_sk_cv_std)"
    ),
    md(
        "## Ridge/Lasso: mean CV RMSE vs. alpha\n\n"
        "The CV analogue of notebook 2's `semilogx` line plot -- error "
        "bars are the std across the 5 folds, the direct visual signal "
        "of how noisy each alpha's score is."
    ),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n\n"
        'axes[0].errorbar(\n'
        "    alphas_gd, ridge_gd_cv_mean, yerr=ridge_gd_cv_std,\n"
        '    marker="o", capsize=3, label="RidgeGD",\n'
        ")\n"
        'axes[0].errorbar(\n'
        "    alphas_sk, ridge_sk_cv_mean, yerr=ridge_sk_cv_std,\n"
        '    marker="o", capsize=3, label="sklearn Ridge",\n'
        ")\n"
        'axes[0].set_xscale("log")\n'
        'axes[0].set_xlabel("alpha")\n'
        'axes[0].set_ylabel("mean CV RMSE (log space)")\n'
        'axes[0].set_title("Ridge: mean CV RMSE vs. alpha")\n'
        "axes[0].legend()\n\n"
        'axes[1].errorbar(\n'
        "    alphas_gd, lasso_gd_cv_mean, yerr=lasso_gd_cv_std,\n"
        '    marker="o", capsize=3, label="LassoGD",\n'
        ")\n"
        'axes[1].errorbar(\n'
        "    alphas_gd, lasso_sk_cv_mean, yerr=lasso_sk_cv_std,\n"
        '    marker="o", capsize=3, label="sklearn Lasso",\n'
        ")\n"
        'axes[1].set_xscale("log")\n'
        'axes[1].set_xlabel("alpha")\n'
        'axes[1].set_ylabel("mean CV RMSE (log space)")\n'
        'axes[1].set_title("Lasso: mean CV RMSE vs. alpha")\n'
        "axes[1].legend()\n\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "## ElasticNet: 5-fold CV over (alpha, l1_ratio)\n\n"
        "Same 5x5 grid as `3_elastic_net.ipynb`."
    ),
    code(
        "alphas_en = np.logspace(-3, 1, 5)\n"
        "l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]\n\n"
        "en_gd_cv_mean = np.zeros((len(alphas_en), len(l1_ratios)))\n"
        "en_sk_cv_mean = np.zeros((len(alphas_en), len(l1_ratios)))\n\n"
        "for i, a in enumerate(alphas_en):\n"
        "    for j, r in enumerate(l1_ratios):\n"
        "        scores = cv_scores(\n"
        "            ElasticNetGD,\n"
        "            dict(\n"
        "                learning_rate=0.01, batch_size=32, n_epochs=300,\n"
        "                alpha=a, l1_ratio=r, random_state=42,\n"
        "            ),\n"
        "            cv_pool_raw, y_cv_pool,\n"
        "        )\n"
        "        en_gd_cv_mean[i, j] = scores.mean()\n\n"
        "        scores = cv_scores(\n"
        "            ElasticNet, dict(alpha=a, l1_ratio=r), cv_pool_raw, y_cv_pool,\n"
        "        )\n"
        "        en_sk_cv_mean[i, j] = scores.mean()"
    ),
    md(
        "## ElasticNet: mean CV RMSE heatmap\n\n"
        "The CV analogue of notebook 3's heatmap."
    ),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n\n"
        "for ax, data, title in [\n"
        '    (axes[0], en_gd_cv_mean, "ElasticNetGD"),\n'
        '    (axes[1], en_sk_cv_mean, "sklearn ElasticNet"),\n'
        "]:\n"
        '    im = ax.imshow(data, aspect="auto", cmap="viridis_r")\n'
        "    ax.set_xticks(range(len(l1_ratios)))\n"
        "    ax.set_xticklabels(l1_ratios)\n"
        "    ax.set_yticks(range(len(alphas_en)))\n"
        '    ax.set_yticklabels([f"{a:.4g}" for a in alphas_en])\n'
        '    ax.set_xlabel("l1_ratio")\n'
        '    ax.set_ylabel("alpha")\n'
        '    ax.set_title(f"{title}: mean CV RMSE")\n'
        "    fig.colorbar(im, ax=ax)\n\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "## CV-selected vs. single-split-selected hyperparameters\n\n"
        "The single-split values are hardcoded from "
        "`2_regularized_regression.ipynb`/`3_elastic_net.ipynb`'s own "
        "executed output (not re-derived) -- the point of this table is "
        "to see, per model, whether 5-fold CV agrees with the earlier "
        "single-split pick or found a different optimum."
    ),
    code(
        "best_ridge_gd_alpha = alphas_gd[np.argmin(ridge_gd_cv_mean)]\n"
        "best_ridge_sk_alpha = alphas_sk[np.argmin(ridge_sk_cv_mean)]\n"
        "best_lasso_gd_alpha = alphas_gd[np.argmin(lasso_gd_cv_mean)]\n"
        "best_lasso_sk_alpha = alphas_gd[np.argmin(lasso_sk_cv_mean)]\n\n"
        "best_en_gd_idx = np.unravel_index(np.argmin(en_gd_cv_mean), en_gd_cv_mean.shape)\n"
        "best_en_sk_idx = np.unravel_index(np.argmin(en_sk_cv_mean), en_sk_cv_mean.shape)\n"
        "best_en_gd_alpha, best_en_gd_l1_ratio = (\n"
        "    alphas_en[best_en_gd_idx[0]], l1_ratios[best_en_gd_idx[1]],\n"
        ")\n"
        "best_en_sk_alpha, best_en_sk_l1_ratio = (\n"
        "    alphas_en[best_en_sk_idx[0]], l1_ratios[best_en_sk_idx[1]],\n"
        ")\n\n"
        "comparison_df = pd.DataFrame(\n"
        "    [\n"
        '        {"model": "RidgeGD", "single-split alpha": 0.03162, "CV alpha": best_ridge_gd_alpha},\n'
        '        {"model": "sklearn Ridge", "single-split alpha": 31.62, "CV alpha": best_ridge_sk_alpha},\n'
        '        {"model": "LassoGD", "single-split alpha": 0.01, "CV alpha": best_lasso_gd_alpha},\n'
        '        {"model": "sklearn Lasso", "single-split alpha": 0.003162, "CV alpha": best_lasso_sk_alpha},\n'
        "        {\n"
        '            "model": "ElasticNetGD", "single-split alpha": 0.01,\n'
        '            "single-split l1_ratio": 0.7, "CV alpha": best_en_gd_alpha,\n'
        '            "CV l1_ratio": best_en_gd_l1_ratio,\n'
        "        },\n"
        "        {\n"
        '            "model": "sklearn ElasticNet", "single-split alpha": 0.01,\n'
        '            "single-split l1_ratio": 0.3, "CV alpha": best_en_sk_alpha,\n'
        '            "CV l1_ratio": best_en_sk_l1_ratio,\n'
        "        },\n"
        "    ]\n"
        ").set_index(\"model\")\n"
        "comparison_df"
    ),
    md(
        "## Refit each model at its CV-selected hyperparameters\n\n"
        "One `build_pipeline()` fit on the entire `cv_pool` (not "
        "per-fold) -- the CV sweep above is only for hyperparameter "
        "selection; the final model is fit the normal way, same as "
        "notebooks 2/3's own \"refit at best alpha\" step."
    ),
    code(
        "pipeline_final = build_pipeline()\n"
        "X_cv_pool = pipeline_final.fit_transform(cv_pool_raw).to_numpy()\n"
        "X_holdout = pipeline_final.transform(holdout_raw).to_numpy()\n\n"
        "ridge_gd_best = RidgeGD(\n"
        "    learning_rate=0.01, batch_size=32, n_epochs=300,\n"
        "    alpha=best_ridge_gd_alpha, random_state=42, verbose=True,\n"
        ")\n"
        "ridge_gd_best.fit(X_cv_pool, y_cv_pool)\n\n"
        "ridge_sk_best = Ridge(alpha=best_ridge_sk_alpha)\n"
        "ridge_sk_best.fit(X_cv_pool, y_cv_pool)\n\n"
        "lasso_gd_best = LassoGD(\n"
        "    learning_rate=0.01, batch_size=32, n_epochs=300,\n"
        "    alpha=best_lasso_gd_alpha, random_state=42, verbose=True,\n"
        ")\n"
        "lasso_gd_best.fit(X_cv_pool, y_cv_pool)\n\n"
        "lasso_sk_best = Lasso(alpha=best_lasso_sk_alpha)\n"
        "lasso_sk_best.fit(X_cv_pool, y_cv_pool)\n\n"
        "en_gd_best = ElasticNetGD(\n"
        "    learning_rate=0.01, batch_size=32, n_epochs=300,\n"
        "    alpha=best_en_gd_alpha, l1_ratio=best_en_gd_l1_ratio,\n"
        "    random_state=42, verbose=True,\n"
        ")\n"
        "en_gd_best.fit(X_cv_pool, y_cv_pool)\n\n"
        "en_sk_best = ElasticNet(alpha=best_en_sk_alpha, l1_ratio=best_en_sk_l1_ratio)\n"
        "en_sk_best.fit(X_cv_pool, y_cv_pool)\n\n"
        "linreg_gd_baseline = LinearRegressionGD(\n"
        "    learning_rate=0.01, batch_size=32, n_epochs=300, random_state=42,\n"
        "    verbose=True,\n"
        ")\n"
        "linreg_gd_baseline.fit(X_cv_pool, y_cv_pool)\n\n"
        "linreg_sk_baseline = LinearRegression()\n"
        "linreg_sk_baseline.fit(X_cv_pool, y_cv_pool)\n\n"
        "MODELS = [\n"
        '    ("LinearRegressionGD (baseline)", linreg_gd_baseline),\n'
        '    ("sklearn LinearRegression (baseline)", linreg_sk_baseline),\n'
        '    (f"RidgeGD (alpha={best_ridge_gd_alpha:.4g})", ridge_gd_best),\n'
        '    (f"sklearn Ridge (alpha={best_ridge_sk_alpha:.4g})", ridge_sk_best),\n'
        '    (f"LassoGD (alpha={best_lasso_gd_alpha:.4g})", lasso_gd_best),\n'
        '    (f"sklearn Lasso (alpha={best_lasso_sk_alpha:.4g})", lasso_sk_best),\n'
        "    (\n"
        '        f"ElasticNetGD (alpha={best_en_gd_alpha:.4g}, "\n'
        '        f"l1_ratio={best_en_gd_l1_ratio})",\n'
        "        en_gd_best,\n"
        "    ),\n"
        "    (\n"
        '        f"sklearn ElasticNet (alpha={best_en_sk_alpha:.4g}, "\n'
        '        f"l1_ratio={best_en_sk_l1_ratio})",\n'
        "        en_sk_best,\n"
        "    ),\n"
        "]"
    ),
    md(
        "## Metrics table (final holdout)\n\n"
        "RMSE in log space is the metric that actually matters (matches "
        "the Kaggle scoring); RMSE in dollar space (via `expm1`) and "
        "MAPE are for human-readable intuition."
    ),
    code(
        "metrics_rows = []\n"
        "for name, model in MODELS:\n"
        "    preds_log = model.predict(X_holdout)\n"
        "    preds_dollars = np.expm1(preds_log)\n"
        "    actual_dollars = np.expm1(y_holdout)\n"
        "    mape = mean_absolute_percentage_error(actual_dollars, preds_dollars)\n"
        "    metrics_rows.append(\n"
        "        {\n"
        '            "model": name,\n'
        '            "RMSE (log space)": rmse(y_holdout, preds_log),\n'
        '            "RMSE (dollars, intuition only)": rmse(\n'
        "                actual_dollars, preds_dollars\n"
        "            ),\n"
        '            "R^2": r2_score(y_holdout, preds_log),\n'
        '            "MAE (log space)": mean_absolute_error(y_holdout, preds_log),\n'
        '            "MAPE (dollars, % of actual price)": 100 * mape,\n'
        "        }\n"
        "    )\n\n"
        "metrics_df = pd.DataFrame(metrics_rows).set_index(\"model\")\n"
        "metrics_df"
    ),
    md(
        "## Summary\n\n"
        "TODO after execution: fill in whether the final-holdout RMSE "
        "improved over notebooks 2/3's single-split-selected models, and "
        "how often (and by how much) the CV-selected hyperparameter "
        "differed from the single-split choice in `comparison_df` above."
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
with open("notebooks/models/4_cross_validation.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
```

- [ ] **Step 2: Run the script, execute the notebook**

Run:
```bash
python3 /path/to/scratchpad/build_cv_nb.py
cd notebooks/models && uv run --project ../.. jupyter nbconvert --to notebook --execute --inplace 4_cross_validation.ipynb
```
Expected: `[NbConvertApp] Writing ... bytes to 4_cross_validation.ipynb`, no error cells. This notebook does substantially more model fitting than notebooks 1-3 (roughly 430 GD/sklearn fits total across the Ridge/Lasso/ElasticNet CV sweeps) — expect this to take longer than previous notebooks; a multi-minute run is normal, not a sign of a stuck kernel.

- [ ] **Step 3: Verify no errors**

Run:
```python
import json

nb = json.load(open("notebooks/models/4_cross_validation.ipynb"))
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

- [ ] **Step 4: Sanity-check the printed tables and plots**

Open the executed notebook and confirm: (1) both Ridge/Lasso mean-CV-RMSE curves show a visible minimum somewhere inside the swept range, not at a corner; (2) both ElasticNet heatmaps show a visible minimum somewhere inside the 5x5 grid, not at a corner; (3) `comparison_df` has a value in every cell (not NaN) for the models that use it, and the CV-selected alphas are in the same order of magnitude as the single-split values, not wildly different; (4) the final `metrics_df`'s log-space RMSE values are in the same ballpark as notebooks 2/3's own metrics tables (within a small margin, not orders of magnitude worse) — if any of these fail, the swept ranges need revisiting before this task is considered done.

- [ ] **Step 5: Fill in the summary markdown cell**

Replace the "TODO after execution" placeholder text in the final markdown cell with the actual comparison read from the executed `comparison_df` and `metrics_df` — e.g. which models' CV-selected alpha matched, diverged, and by how much, and whether final-holdout RMSE improved over notebooks 2/3. Use `NotebookEdit` on the executed notebook (not the build script) so the real printed values are quoted, then re-run only that markdown cell's neighboring context is unaffected (no re-execution needed since it's a markdown-only edit).

- [ ] **Step 6: Commit**

```bash
git add notebooks/models/4_cross_validation.ipynb
git commit -m "feat: add cross-validated hyperparameter selection for Ridge/Lasso/ElasticNet"
```

---

## Self-Review Notes

- **Spec coverage:** `kfold_split` — from-scratch, index-only, reproducible under a fixed `random_state`, fold-partition/disjointness/size-balance tested (Task 1); notebook — CV pool vs. final holdout split at the same `random_state=42` as notebooks 2/3, leak-free per-fold pipeline refit via `cv_scores`, CV sweeps reusing notebooks 2/3's exact alpha/l1_ratio grids, mean+std aggregation, CV-vs-single-split comparison table, final refit at CV-selected combo, and a `metrics_df` on the untouched holdout (Task 2) — every spec section has a task.
- **Type consistency checked:** `kfold_split`'s signature (`n_samples: int, n_splits: int = 5, random_state: int | None = None -> list[tuple[np.ndarray, np.ndarray]]`) matches how Task 2's `cv_scores` calls it (`kfold_split(len(X_raw), n_splits, random_state)`, unpacking `for train_idx, val_idx in folds`); `cv_scores(model_cls, model_kwargs, X_raw, y, ...)` is called identically for every GD and sklearn model (`model_cls(**model_kwargs)` duck-types across both), matching how `MODELS` is built the same way as notebooks 2/3.
- **No placeholders:** every step has real code and real hyperparameters (`np.logspace(-3, 1, 9)`/`np.logspace(-3, 3, 9)` for Ridge/Lasso, `np.logspace(-3, 1, 5)` x `[0.1, 0.3, 0.5, 0.7, 0.9]` for ElasticNet, and the hardcoded single-split comparison values `0.03162`/`31.62`/`0.01`/`0.003162`/`(0.01, 0.7)`/`(0.01, 0.3)` copied directly from notebooks 2/3's own executed output). The one intentional exception is the final summary markdown cell's placeholder text, which Step 5 explicitly requires filling in with real executed values before the task is done — not left as-is.
