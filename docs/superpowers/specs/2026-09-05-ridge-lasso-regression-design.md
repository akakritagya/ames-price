# Ridge & Lasso Regression (from scratch + sklearn baseline) — Design

**Status:** approved, pending implementation
**Follows:** `docs/superpowers/specs/2026-09-05-linear-regression-design.md`,
which explicitly deferred regularization ("Open questions") pending a
later, separate spec. This is that spec.

## Purpose

Second modeling step: implement Ridge (L2) and Lasso (L1) regularized
linear regression from scratch (mini-batch gradient descent, plain
numpy), then compare each against scikit-learn's own `Ridge`/`Lasso` on
the same data. The pipeline's known rank-deficiency (`rank_ == 207` of
236 one-hot-encoded columns, observed in notebook 1) makes this a natural
next step: regularization is the mechanism that actually addresses it,
rather than relying on sklearn's SVD solver's minimum-norm fallback.

## Non-goals

- No ElasticNet (L1+L2 combined) — Ridge and Lasso cover the two pure
  cases; a combined penalty is a natural but separate later extension.
- No coordinate descent / proximal (ISTA) solver for Lasso — plain
  subgradient descent, matching the project's existing GD style
  (`LinearRegressionGD`) rather than introducing a structurally different
  update rule.
- No cross-validation — reuse the single fixed holdout split from
  notebook 1, not k-fold.
- No attempt to make GD's `alpha` and sklearn's `alpha` numerically
  equivalent. sklearn's `Ridge` and `Lasso` each normalize their internal
  objective differently (and differently from each other), so a shared
  numeric `alpha` value does not imply the same shrinkage strength across
  implementations. Each model family's `alpha` is swept independently;
  comparison happens at the level of best-achievable validation RMSE and
  qualitative sparsity behavior, not identical alpha meaning.
- No Kaggle submission file — same deferral as notebook 1.

## Module layout

```text
src/ames_price/
    models/
        __init__.py
        linear_regression.py        # existing, unchanged
        ridge_gd.py                 # RidgeGD
        lasso_gd.py                 # LassoGD
tests/
    test_linear_regression.py       # existing, unchanged
    test_ridge_gd.py
    test_lasso_gd.py
notebooks/
    models/
        1_linear_regression.ipynb   # existing, unchanged
        2_regularized_regression.ipynb
```

One class per file, mirroring the existing `linear_regression.py`
convention — each file is a complete, standalone explanation of one
algorithm rather than one file branching on a `penalty` parameter.

## Component: `RidgeGD`

```python
# src/ames_price/models/ridge_gd.py
class RidgeGD:
    def __init__(
        self,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        n_epochs: int = 100,
        alpha: float = 1.0,
        random_state: int | None = None,
    ) -> None: ...

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        # sets self.coef_, self.intercept_, self.loss_history_
        ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...
```

Same sklearn-shaped API and same structural skeleton as
`LinearRegressionGD` (zero-init `coef_`, zero-init scalar `intercept_`,
seeded per-epoch shuffle, mini-batch loop, full-training-set loss
recorded once per epoch) — the only difference is what gets added to the
gradient.

**Objective:** `MSE + alpha * sum(coef_ ** 2)`. The intercept is excluded
from the penalty (standard convention, matches sklearn) — regularizing
the intercept would shrink the model's baseline prediction toward zero
for no principled reason, since only the log-price *origin*, not its
relationship to features, is at stake.

**Gradient, per batch:**

```python
error = X_batch @ coef_ + intercept_ - y_batch
grad_coef = (2 / len(batch_idx)) * (X_batch.T @ error) + 2 * alpha * coef_
grad_intercept = (2 / len(batch_idx)) * error.sum()  # unchanged, unpenalized
```

The penalty term (`2 * alpha * coef_`) is **not** scaled by batch size —
it's a property of the coefficients themselves, not the data sample, so
it applies at full strength on every batch regardless of `batch_size`.
This keeps `alpha`'s meaning stable if `batch_size` changes.

**Loss recorded in `loss_history_`:** the full regularized objective
(`MSE + alpha * sum(coef_ ** 2)`), evaluated over the full training set
once per epoch — not bare MSE. This is what the optimizer is actually
minimizing, and the docstring says so explicitly, since it means
`RidgeGD.loss_history_` is not directly comparable to
`LinearRegressionGD.loss_history_` at the same epoch.

## Component: `LassoGD`

```python
# src/ames_price/models/lasso_gd.py
class LassoGD:
    def __init__(
        self,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        n_epochs: int = 100,
        alpha: float = 1.0,
        random_state: int | None = None,
    ) -> None: ...

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        # sets self.coef_, self.intercept_, self.loss_history_
        ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...
```

Identical skeleton to `RidgeGD`/`LinearRegressionGD`. The only difference
is the penalty: `sum(abs(coef_))` instead of `sum(coef_ ** 2)`.

**The non-differentiability at zero:** `abs(coef_)` has no gradient at
`coef_ == 0`. Rather than a proximal/soft-thresholding solver (the
textbook-correct approach, but a structurally different update rule from
plain GD), this uses the simpler subgradient approximation, applied
uniformly with `RidgeGD`'s and `LinearRegressionGD`'s same "compute
gradient, take a step" shape:

```python
error = X_batch @ coef_ + intercept_ - y_batch
grad_coef = (2 / len(batch_idx)) * (X_batch.T @ error) + alpha * np.sign(coef_)
grad_intercept = (2 / len(batch_idx)) * error.sum()  # unchanged, unpenalized
```

`np.sign(0) == 0`, so a coefficient sitting exactly at zero gets no
penalty push in either direction that batch — it can still move if the
MSE gradient term pulls it away, but the penalty itself won't be what
moves it off zero. This means `LassoGD`'s sparsity is softer than
proximal Lasso's exact zeroing: coefficients shrink *toward* zero and
some will land very close to it, but "near zero" (checked with a
tolerance), not bitwise `0.0`, is the correct thing to assert or expect.

**Loss recorded in `loss_history_`:** `MSE + alpha * sum(abs(coef_))`,
same reasoning as `RidgeGD`.

## Testing strategy

Same synthetic-fixture approach as `test_linear_regression.py` — no real
Ames data in unit tests; `2_regularized_regression.ipynb` is where
real-data behavior is demonstrated and visually checked.

**`tests/test_ridge_gd.py`:**
- On the same known-linear synthetic dataset as `LinearRegressionGD`
  (`y = 3*x1 - 2*x2 + 5 + small_noise`), `fit` converges to
  *approximately* the true weights and intercept — a **looser** tolerance
  than `LinearRegressionGD`'s test, since Ridge is intentionally biased
  toward smaller coefficients and won't recover them exactly even with
  low noise and many epochs.
- `predict`'s output shape matches the number of input rows.
- `loss_history_` has exactly `n_epochs` entries and is non-increasing
  overall (the regularized objective, not bare MSE).

**`tests/test_lasso_gd.py`:**
- A synthetic dataset with a known linear relationship where *some*
  features have a true coefficient of exactly zero (e.g.
  `y = 3*x1 - 2*x2 + 0*x3 + 0*x4 + 5 + small_noise`) and others don't.
  After `fit` with a large-enough `alpha`, the true-zero features'
  learned coefficients are close to zero (within a tolerance), while
  `x1`/`x2`'s coefficients remain close to their true nonzero values —
  the test that actually demonstrates Lasso's defining behavior
  (feature selection), not just "it fits."
- `predict`'s output shape matches the number of input rows.
- `loss_history_` has exactly `n_epochs` entries and is non-increasing
  overall.

## Data flow (`notebooks/models/2_regularized_regression.ipynb`)

1. Load `train.csv`; drop outlier `Id` 524/1299; compute
   `log_sale_price` — same as notebook 1.
2. Split raw features/target first, `fit_transform` `build_pipeline()` on
   the train split only, `transform` the validation split — the
   leak-free order notebook 1 was corrected to use (never
   `fit_transform` before splitting).
3. **Alpha sweep:** for a log-spaced range (e.g. `np.logspace(-3, 1, 9)`,
   9 values from 0.001 to 10), fit `RidgeGD(alpha=a)`, sklearn
   `Ridge(alpha=a)`, `LassoGD(alpha=a)`, and sklearn `Lasso(alpha=a)` on
   the train split, and record each model's validation-split RMSE (log
   space). Plot validation RMSE vs. alpha (`plt.semilogx`), one line per
   model, in two subplots (Ridge pair, Lasso pair) — visualizes the
   bias-variance tradeoff and each family's best-performing region.
4. Pick the best (lowest validation RMSE) alpha independently for each of
   the 4 models; refit each at its own best alpha.
5. Report a metrics table (RMSE log space, RMSE dollar space via
   `expm1`, R², MAE, MAPE) for all 4 regularized models side by side,
   with notebook 1's plain `LinearRegressionGD`/sklearn `LinearRegression`
   numbers included as a reference baseline (recomputed inline, not
   hand-copied — reuses `build_pipeline()` the same way).
6. Coefficient-sparsity comparison: for each of the 4 regularized models
   at its best alpha, count/plot how many coefficients fall below a
   small-magnitude threshold (e.g. `abs(coef_) < 1e-3`) — expected to
   show Lasso (both GD and sklearn) driving meaningfully more
   coefficients toward zero than Ridge, directly illustrating Lasso's
   feature-selection property against the pipeline's 236-column,
   rank-207 design.

## Open questions (explicitly out of scope for this round)

- ElasticNet (combined L1+L2 penalty).
- A proximal/coordinate-descent solver for exact Lasso sparsity.
- Cross-validation for alpha selection (this round uses a single
  train/validation split, same as the alpha sweep's holdout).
- Kaggle submission / scoring against `test.csv`.
