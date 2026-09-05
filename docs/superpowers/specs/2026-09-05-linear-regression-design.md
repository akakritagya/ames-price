# Linear Regression (from scratch + sklearn baseline) — Design

**Status:** approved, pending implementation
**Follows:** the completed preprocessing/feature-engineering pipeline
(`docs/superpowers/specs/2026-09-04-preprocessing-feature-engineering-design.md`),
specifically its "Open questions" notes on scaling and rank-deficiency —
both get resolved here, in the model's own scope rather than the
pipeline's.

## Purpose

First modeling step on the Ames Housing data: implement ordinary linear
regression from scratch (mini-batch gradient descent, plain numpy) to
understand the mechanics, then compare it against scikit-learn's own
`LinearRegression` on the same data. This is a learning project, so the
emphasis is on explaining *why* the gradient descent loop works, not on
squeezing out leaderboard score.

## Non-goals

- No regularization (Ridge/Lasso). Plain OLS via gradient descent, as
  decided — the pipeline's known rank-deficiency (one-hot columns, no
  `drop="first"`) is left as an observed side effect, not fixed here.
  Regularized models are natural candidates for a later, separate spec.
- No cross-validation. A single fixed holdout split, not k-fold — enough
  to compare two models meaningfully; CV is a later refinement once
  there's a reason to pick between more than two candidates.
- No Kaggle submission file in this round. `test.csv` has no labels;
  scoring against it is a separate, later step once a model is actually
  chosen.
- No hyperparameter search. `learning_rate`/`batch_size`/`n_epochs` are
  picked by eye from the loss curve, not swept.

## Module layout

```
src/ames_price/
    pipeline.py                     # +1 step: ("scale", StandardScaler())
    models/
        __init__.py
        linear_regression.py        # LinearRegressionGD
tests/
    test_linear_regression.py
notebooks/
    models/
        1_linear_regression.ipynb   # fits + compares both models on real data
```

## Pipeline change: scaling

Gradient descent needs comparable feature scales to converge well —
`build_pipeline()`'s current output mixes log1p'd areas, small ordinal
integers, and 0/1 one-hot flags. A `StandardScaler` step is appended as
the 4th stage:

```python
# src/ames_price/pipeline.py
def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("impute", Imputer()),
            ("engineer", FeatureEngineer()),
            ("encode", Encoder()),
            ("scale", StandardScaler()),
        ]
    )
```

Used directly from `sklearn.preprocessing` — no custom transformer, since
only the *model* is the from-scratch part of this work, matching the
project's existing use of `sklearn.base` machinery for `Imputer`/
`FeatureEngineer`/`Encoder`. Fit on train only, applied to test the same
way as every other pipeline step (no leakage). Scaling one-hot 0/1
columns is harmless (they just get centered/scaled like anything else)
and keeps this one pipeline shared by both models in the comparison.

## Component: `LinearRegressionGD`

```python
# src/ames_price/models/linear_regression.py
class LinearRegressionGD:
    def __init__(
        self,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        n_epochs: int = 100,
        random_state: int | None = None,
    ) -> None: ...

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        # sets self.coef_, self.intercept_, self.loss_history_
        ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...
```

sklearn-shaped API (`fit`/`predict`, `coef_`/`intercept_` attributes) so
the comparison notebook can run this model and sklearn's own
`LinearRegression` through identical calling code.

**Initialization:** `coef_` starts at zero (one weight per feature),
`intercept_` starts at zero, tracked as its own scalar attribute rather
than by augmenting `X` with a column of ones — keeps each parameter's
gradient update an explicit, separately-explainable line rather than one
matrix trick.

**Loss:** mean squared error, `mean((y_pred - y) ** 2)`, over the current
batch. `y_pred = X @ coef_ + intercept_`.

**Mini-batch loop:** for each of `n_epochs` epochs — shuffle training row
indices (seeded by `random_state`, so a given construction is
reproducible), then step through non-overlapping slices of `batch_size`
rows, computing the batch's MSE gradient with respect to `coef_` and
`intercept_` and updating both by `learning_rate * gradient`. After each
epoch, evaluate the loss over the *full* training set (not just the last
batch) and append it to `loss_history_` — this is what the notebook plots
as the convergence curve.

**Stopping:** fixed `n_epochs`, no tolerance-based early stopping. For a
first simple model, picking `n_epochs` by eye from where the loss curve
flattens is the intended workflow, not an automated criterion.

**Edge cases, deliberately unhandled:** `batch_size > n_samples` (numpy
slicing degrades gracefully to one full-data batch — effectively batch
GD); mismatched/non-2D shapes (let numpy's own error surface); a
diverging `learning_rate` producing `NaN` losses (visible directly in the
plotted loss curve — more instructive here than a silent guard). None of
these get explicit validation code, per the project's "don't validate
scenarios that can't happen / aren't the point" convention.

## Target handling

`log_sale_price = np.log1p(SalePrice)` is computed inline in the notebook
at load time — not added to `constants.py`, since it's a one-line
transform used only here, not a reusable pipeline rule. Both models train
and evaluate in log space (matching the actual Kaggle scoring metric,
RMSE of log predictions vs. log actuals). `np.expm1` is applied to
predictions only when reporting a dollar-scale RMSE for human-readable
sanity-checking — that number is never used to pick between models.

## Data flow (`notebooks/models/1_linear_regression.ipynb`)

1. Load `train.csv`; drop outlier `Id` 524/1299 (per the preprocessing
   design's outlier-handling note).
2. `y = np.log1p(df_train["SalePrice"])`.
3. `X = build_pipeline().fit_transform(df_train[feature_cols])`.
4. Single holdout split, 80/20, fixed `random_state`.
5. Fit `LinearRegressionGD()` and sklearn's `LinearRegression()` on the
   same training split.
6. Predict on the holdout split with both.
7. Report, for both models: RMSE in log space (the real metric), RMSE in
   dollar space via `expm1` (intuition only), R².
8. Plot `LinearRegressionGD`'s `loss_history_` (the convergence curve).
9. Brief discussion comparing the two models' coefficients/predictions —
   how close does gradient descent get to sklearn's closed-form solution.

## Testing strategy

`tests/test_linear_regression.py`, on a small synthetic dataset with a
known linear relationship (e.g. `y = 3*x1 - 2*x2 + 5 + small_noise`), not
the real 237-column data — mirrors how the preprocessing pipeline's unit
tests use synthetic fixtures while `preprocessing_check.ipynb` covers
real-data behavior separately.

- Given enough epochs/rows on the synthetic problem, `fit` converges
  close to the true weights (`[3, -2]`) and intercept (`5`), within a
  loose tolerance.
- `predict`'s output shape matches the number of input rows.
- `loss_history_` has exactly `n_epochs` entries and is non-increasing
  overall on this well-conditioned, low-noise synthetic problem.

No test runs the real Ames data or asserts a specific score —
`notebooks/models/1_linear_regression.ipynb` is where real-data behavior
is demonstrated and visually checked, matching the project's existing
notebook-verifies/tests-cover-rules split.

## Open questions (explicitly out of scope for this round)

- Regularization (Ridge/Lasso) to address the pipeline's known
  rank-deficiency — left for a later model.
- Cross-validation / hyperparameter search for `learning_rate`,
  `batch_size`, `n_epochs`.
- Kaggle submission / scoring against `test.csv`.
- Whether `StandardScaler` should be swapped for something that respects
  the one-hot columns differently (e.g. not scaling already-0/1 columns)
  — left as the simplest working option unless a later model shows it
  matters.
