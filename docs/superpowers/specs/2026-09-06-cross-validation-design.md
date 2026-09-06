# Cross-Validated Hyperparameter Selection — Design

**Status:** approved, pending implementation
**Follows:** `docs/superpowers/specs/2026-09-06-elastic-net-regression-design.md`,
which explicitly flagged CV as "the next, separate round of work"
(Non-goals): "Cross-validation for `(alpha, l1_ratio)` selection ...
planned as the next round of work." This is that round.

## Purpose

Notebooks 2 and 3 pick each regularized model's `alpha` (or
`alpha, l1_ratio`) by sweeping a grid and reading off whichever value
scores best on a single, fixed train/validation split. That's one noisy
estimate per candidate — it can't tell the difference between "this
alpha is genuinely best" and "this alpha got lucky on this particular
20% of rows." This round replaces the single-split sweep with 5-fold
cross-validation for every regularized model (`RidgeGD`, sklearn
`Ridge`, `LassoGD`, sklearn `Lasso`, `ElasticNetGD`, sklearn
`ElasticNet`), and explicitly compares the CV-selected hyperparameter
against notebooks 2/3's single-split choice for each model — the
payoff being visibility into whether that earlier choice was robust.

## Non-goals

- No nested CV and no automated significance test for "did the CV
  choice differ from the single-split choice" — the comparison is a
  side-by-side table/plot read by eye, same rigor level as this
  project's other model-comparison notebooks.
- No parallelization of the fold loops. The dataset is small (~1166
  training-pool rows) and GD fits are fast (verified in notebooks 1-3);
  expected to run in a couple of minutes end to end.
- No new alpha/l1_ratio grids — reuses notebook 2's `alphas_gd` (9
  points) / `alphas_sk` (9 points, wider range) for Ridge/Lasso, and
  notebook 3's 5x5 `alphas` x `l1_ratios` grid for ElasticNet, so this
  notebook is directly comparable to the earlier ones rather than
  introducing a new resolution to reason about.
- No Kaggle submission file — same deferral as notebooks 1-3.
- `LinearRegressionGD`/sklearn `LinearRegression` are refit once as an
  unregularized reference point, same as notebooks 2/3, but are **not**
  cross-validated — they have no penalty hyperparameter to select.

## Module layout

```text
src/ames_price/
    model_selection.py           # kfold_split()
    models/                      # existing, unchanged
        linear_regression.py
        ridge_gd.py
        lasso_gd.py
        elastic_net_gd.py
tests/
    test_model_selection.py
notebooks/
    models/
        1_linear_regression.ipynb       # existing, unchanged
        2_regularized_regression.ipynb  # existing, unchanged
        3_elastic_net.ipynb             # existing, unchanged
        4_cross_validation.ipynb
```

`model_selection.py` sits alongside `pipeline.py` at the package root
(not under `models/`), since it's a data-splitting utility, not a model.

## Component: `kfold_split`

```python
# src/ames_price/model_selection.py
def kfold_split(
    n_samples: int, n_splits: int = 5, random_state: int | None = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Shuffle 0..n_samples-1, then split into n_splits folds.

    Returns a list of (train_idx, val_idx) pairs, one per fold: for
    each fold, val_idx is that fold's held-out indices and train_idx is
    every other index. Folds partition the full index range exactly
    once (sizes differ by at most 1 row when n_samples isn't evenly
    divisible by n_splits -- the first n_samples % n_splits folds get
    one extra row).
    """
```

Same shuffle style as the GD models: `np.random.default_rng(random_state)`
then `rng.permutation(n_samples)`, so a fixed `random_state` reproduces
the same folds. This is index-only — it never touches `X`/`y` or the
pipeline, matching how `train_test_split` is used elsewhere in this
project (split first, transform per-split after).

**`tests/test_model_selection.py`:**

- Folds partition all indices exactly once: concatenating every fold's
  `val_idx` and sorting reproduces `range(n_samples)` with no
  duplicates or gaps.
- `train_idx`/`val_idx` are disjoint within every fold.
- Fold sizes are as equal as possible: for `n_samples=23, n_splits=5`,
  val fold sizes are `[5, 5, 5, 4, 4]` (or some permutation), never
  differing by more than 1.
- Same `random_state` produces identical folds across two calls;
  `random_state=None` is not asserted for inequality (non-reproducible
  by design, same as the GD models' own `random_state=None` behavior).

## Data flow (`notebooks/models/4_cross_validation.ipynb`)

1. Load `train.csv`; drop outlier `Id` 524/1299; compute
   `log_sale_price`. Split raw features/target into an 80% **CV pool**
   and a 20% **final holdout** via `train_test_split(test_size=0.2,
   random_state=42)` — the same split notebooks 2/3 use for their own
   train/validation split, so this notebook's final holdout metrics are
   directly comparable to theirs.
2. **CV sweep**, per model family:
   - For each candidate `alpha` (Ridge/Lasso) or `(alpha, l1_ratio)`
     (ElasticNet): for each of `kfold_split(len(cv_pool), n_splits=5,
     random_state=42)`'s 5 folds, build a **fresh** `build_pipeline()`,
     `fit_transform` on that fold's training rows, `transform` on its
     held-out rows, fit the model on the transformed training rows,
     score validation RMSE (log space) on the transformed held-out
     rows. A fresh pipeline per fold keeps `StandardScaler`'s
     mean/std leak-free — they must never see a fold's held-out rows,
     even indirectly through the CV pool's aggregate statistics.
   - Aggregate each combo's 5 fold RMSEs to a mean and std.
3. **Plots:** mean CV RMSE vs. alpha with error bars (std) for
   Ridge/Lasso (the CV analogue of notebook 2's `semilogx` line plot);
   a mean-CV-RMSE heatmap (alpha x l1_ratio) for ElasticNet (the CV
   analogue of notebook 3's heatmap).
4. **Selection:** pick each model's best combo by lowest mean CV RMSE.
   Build a comparison table: CV-selected value vs. notebook 2/3's
   single-split-selected value, per model — did they agree?
5. **Final refit:** refit each of the 6 regularized models, plus the
   unregularized `LinearRegressionGD`/sklearn `LinearRegression`
   baseline, once each on the **entire CV pool** (one `build_pipeline()`
   fit, not per-fold) at its selected combo — same "refit at best
   alpha" step as notebooks 2/3.
6. **Metrics table:** evaluate all 8 refit models on the untouched
   final holdout, reported as a `metrics_df` (RMSE log/dollar space,
   R^2, MAE, MAPE) — same DataFrame pattern as notebooks 1-3.
7. Summary: does the final holdout RMSE meaningfully improve over
   notebooks 2/3's single-split-selected models, and — the main point
   of this round — how often (and by how much) did the CV-selected
   hyperparameter differ from the single-split choice?

## Open questions (explicitly out of scope for this round)

- Nested CV or nested nested holdout for an unbiased estimate of the
  CV-selection procedure itself.
- Automated (non-visual) significance testing between CV-selected and
  single-split-selected hyperparameters.
- Kaggle submission / scoring against `test.csv`.
