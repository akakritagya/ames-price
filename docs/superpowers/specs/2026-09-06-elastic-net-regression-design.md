# ElasticNet Regression (from scratch + sklearn baseline) — Design

**Status:** approved, pending implementation
**Follows:** `docs/superpowers/specs/2026-09-05-ridge-lasso-regression-design.md`,
which explicitly flagged ElasticNet as "a natural but separate later
extension" (Non-goals). This is that extension.

## Purpose

Third modeling step: implement ElasticNet (combined L1+L2) regularized
linear regression from scratch (mini-batch gradient descent, plain
numpy), then compare it against scikit-learn's own `ElasticNet` on the
same data, alongside the existing plain/Ridge/Lasso baselines. Ridge and
Lasso each address the pipeline's rank-deficiency (`rank_ == 207` of 236
one-hot-encoded columns) in different ways (shrinkage vs. sparsity);
ElasticNet's mixed penalty is the natural third point on that spectrum.

## Non-goals

- No coordinate descent / proximal (ISTA) solver — plain subgradient
  descent, matching `RidgeGD`/`LassoGD`'s existing GD style rather than
  introducing a structurally different update rule.
- No cross-validation — reuse a single fixed holdout split, same as the
  Ridge/Lasso round. Proper CV/hyperparameter tuning across all
  regularized models is the next, separate round of work.
- No attempt to make `ElasticNetGD`'s `alpha`/`l1_ratio` numerically
  match sklearn's `ElasticNet`, or `RidgeGD`/`LassoGD` at the
  `l1_ratio=0`/`l1_ratio=1` edges beyond the one deliberate exception
  noted below (`l1_ratio=1` matches `LassoGD` exactly). Each model's
  `alpha` is swept independently; comparison happens at the level of
  best-achievable validation RMSE and qualitative sparsity behavior.
- No Kaggle submission file — same deferral as notebooks 1 and 2.

## Module layout

```text
src/ames_price/
    models/
        __init__.py
        linear_regression.py        # existing, unchanged
        ridge_gd.py                 # existing, unchanged
        lasso_gd.py                 # existing, unchanged
        elastic_net_gd.py           # ElasticNetGD
tests/
    test_linear_regression.py       # existing, unchanged
    test_ridge_gd.py                # existing, unchanged
    test_lasso_gd.py                # existing, unchanged
    test_elastic_net_gd.py
notebooks/
    models/
        1_linear_regression.ipynb       # existing, unchanged
        2_regularized_regression.ipynb  # existing, unchanged
        3_elastic_net.ipynb
```

One class per file, continuing the established convention.

## Component: `ElasticNetGD`

```python
# src/ames_price/models/elastic_net_gd.py
class ElasticNetGD:
    def __init__(
        self,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        n_epochs: int = 100,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        random_state: int | None = None,
    ) -> None: ...

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        # sets self.coef_, self.intercept_, self.loss_history_
        ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...
```

Same sklearn-shaped API and same structural skeleton as
`LinearRegressionGD`/`RidgeGD`/`LassoGD` (zero-init `coef_`, zero-init
scalar `intercept_`, seeded per-epoch shuffle, mini-batch loop,
full-training-set loss recorded once per epoch) — the only difference is
the combined penalty term added to the coefficient gradient.

**Objective:**
`MSE + alpha * l1_ratio * sum(abs(coef_)) + alpha * 0.5 * (1 - l1_ratio) * sum(coef_ ** 2)`,
matching sklearn's `ElasticNet` parameterization: `alpha` is the overall
penalty strength, `l1_ratio` in `[0, 1]` is the mixing weight between the
L1 and L2 terms. The intercept is excluded from the penalty, same
reasoning as `RidgeGD`/`LassoGD`.

**Gradient, per batch:**

```python
error = X_batch @ coef_ + intercept_ - y_batch
grad_coef = (2 / len(batch_idx)) * (X_batch.T @ error) + alpha * (
    l1_ratio * np.sign(coef_) + (1 - l1_ratio) * coef_
)
grad_intercept = (2 / len(batch_idx)) * error.sum()  # unchanged, unpenalized
```

As with `RidgeGD`/`LassoGD`, the penalty gradient term is **not** scaled
by batch size — it applies at full strength on every batch regardless of
`batch_size`, keeping `alpha`'s meaning stable if `batch_size` changes.

**Relationship to `LassoGD`/`RidgeGD` at the edges:** at `l1_ratio=1`,
the penalty gradient reduces to `alpha * np.sign(coef_)` — bitwise
identical to `LassoGD` at the same `alpha`. At `l1_ratio=0`, it reduces
to `alpha * coef_`, which is **not** numerically equal to `RidgeGD`'s
`2 * alpha * coef_` at the same `alpha` (a 4x difference in effective L2
strength for the same nominal `alpha`, because `RidgeGD`'s objective
omits the `0.5` factor sklearn's `ElasticNet`/`Ridge` convention
includes). This is documented explicitly in the class docstring as
expected behavior, not a bug — consistent with the project's existing
stance that `alpha` is not comparable 1:1 across model families.

**Loss recorded in `loss_history_`:** the full regularized objective
above, evaluated over the full training set once per epoch — not bare
MSE, same convention as `RidgeGD`/`LassoGD`.

## Testing strategy

Same synthetic-fixture approach as `test_lasso_gd.py` — no real Ames
data in unit tests; `3_elastic_net.ipynb` is where real-data behavior is
demonstrated and visually checked.

**`tests/test_elastic_net_gd.py`:**

- Reuses `LassoGD`'s sparse synthetic fixture
  (`y = 3*x1 - 2*x2 + 0*x3 + 0*x4 + 5 + small_noise`, `x3`/`x4` true
  coefficient exactly zero).
- **`l1_ratio=1.0` matches `LassoGD`'s own test:** with the same `alpha`
  as `test_lasso_gd.py`'s sparsity test, assert the same behavior —
  `x3`/`x4` near zero, `x1`/`x2` close to true values, intercept tight.
  This is a direct cross-check that the pure-L1 edge case behaves like
  `LassoGD` in spirit (not bitwise-identical output, since it's a
  separate GD run, but the same qualitative behavior at a similarly
  loose tolerance).
- **`l1_ratio=0.5` shows mixed behavior:** a looser-tolerance recovery
  check — `x1`/`x2` still recognizably close to their true values, but
  with looser tolerance than the `l1_ratio=1.0` case (the added L2 term
  biases all coefficients toward zero, not just the true-zero ones), and
  `x3`/`x4` shrunk toward zero but not necessarily as tightly as the
  pure-L1 case. This demonstrates the "mix" behavior rather than just
  re-testing an edge case.
- `predict`'s output shape matches the number of input rows.
- `loss_history_` has exactly `n_epochs` entries and is non-increasing
  overall (full-batch GD, same reasoning as `RidgeGD`/`LassoGD`'s
  equivalent tests — the combined penalty only makes the objective more
  convex, never less).

## Data flow (`notebooks/models/3_elastic_net.ipynb`)

1. Load `train.csv`; drop outlier `Id` 524/1299; compute
   `log_sale_price`; split raw features/target first, `fit_transform`
   `build_pipeline()` on the train split only, `transform` the
   validation split — same leak-free setup as notebooks 1 and 2.
2. **Grid sweep:** for `alphas = np.logspace(-3, 1, 5)` ×
   `l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]` (25 combinations — narrower
   than Ridge/Lasso's 9-point 1D sweep since this is now 2D), fit
   `ElasticNetGD(alpha=a, l1_ratio=r)` and sklearn
   `ElasticNet(alpha=a, l1_ratio=r)` on the train split, and record each
   model's validation-split RMSE (log space).
3. **Heatmap** (alpha × l1_ratio) of validation RMSE, one subplot for
   `ElasticNetGD` and one for sklearn `ElasticNet` — the 2D analogue of
   notebook 2's `semilogx` line plot, visualizing where each model's best
   region sits in the combined hyperparameter space.
4. Pick the best (lowest validation RMSE) `(alpha, l1_ratio)` combo
   independently for each of the 2 models; refit each at its own best
   combo.
5. Report a metrics table (RMSE log space, RMSE dollar space via
   `expm1`, R², MAE, MAPE) for both ElasticNet models side by side with
   the full existing roster from notebook 2 (plain baseline, Ridge,
   Lasso — GD and sklearn each, recomputed inline via `build_pipeline()`,
   not hand-copied).
6. Coefficient-sparsity comparison, extending notebook 2's bar chart to
   include `ElasticNetGD`/sklearn `ElasticNet` — expected to show
   ElasticNet's sparsity sitting between Ridge (least sparse) and Lasso
   (most sparse), directly illustrating the mixed penalty's intermediate
   feature-selection behavior.

## Open questions (explicitly out of scope for this round)

- A proximal/coordinate-descent solver for exact sparsity.
- Cross-validation for `(alpha, l1_ratio)` selection (this round uses a
  single train/validation split, same as the grid sweep's holdout) —
  planned as the next round of work.
- Kaggle submission / scoring against `test.csv`.
