# Preprocessing & Feature Engineering — Design

**Status:** approved, pending implementation
**Follows:** the completed EDA series (`notebooks/eda/0-7`), specifically
`7_wrap_up.ipynb`'s "Consolidated action list" and "Not yet decided"
sections.

## Purpose

Turn the EDA findings into reusable, tested code that takes raw
`train.csv`/`test.csv` to a fully numeric, model-ready matrix — with no
leakage between train and test, and every EDA judgment call resolved to a
concrete, justified rule rather than a vague recommendation.

Two pieces, split by the nature of the decision:

- **Preprocessing** (mechanical, directly dictated by EDA): missingness
  fills, ordinal/nominal encoding, log transforms, unseen-category
  handling.
- **Feature engineering** (judgment calls EDA flagged but left open):
  structural-identity resolution, derived features, rare-level bucketing,
  column drops.

Outlier row-handling (`Id` 524/1299) is explicitly **outside** both
pieces — see "Outlier handling" below.

## Non-goals

- No modeling code. This produces a training matrix; fitting an actual
  model is a separate, later piece of work.
- No new EDA. Every rule below traces back to a specific finding already
  on record in `notebooks/eda/`; this doc doesn't introduce new analysis,
  only resolves things EDA explicitly left as open questions.
- Rare-level bucketing is scoped to the columns EDA explicitly flagged as
  ready to act on. Columns EDA said needed more care before bucketing
  (`Neighborhood`, `Exterior1st`/`2nd`, `Condition1`, `RoofStyle`,
  `SaleType`, `Foundation` — all noted in `4_bivariate.ipynb` as "worth
  encoding carefully rather than dropping tail levels blind") are left
  untouched in this round.

## Module layout

```
src/ames_price/
    preprocessing.py   # Imputer, Encoder
    features.py         # FeatureEngineer
    pipeline.py          # build_pipeline() -> Pipeline([impute, engineer, encode])
tests/
    test_preprocessing.py
    test_features.py
    test_pipeline.py
notebooks/
    preprocessing_check.ipynb   # runs the pipeline, visually verifies the output
```

All three transformers are `sklearn.base.BaseEstimator` +
`TransformerMixin` subclasses: `fit(X, y=None)` learns any statistic from
training data only, `transform(X)` applies it. This is what makes reusing
the exact same fitted state on `df_test` safe and leak-free, and is why
`scikit-learn` (already a dependency) is used directly rather than
hand-rolled functions.

## Component 1: `Imputer` (preprocessing)

Fills every `NaN` so the frame that leaves this step has none. Reuses
`2_missingness.ipynb`'s structural-vs-real-gap classification, and its
finding that "structural" isn't perfectly uniform per column — a few
named rows have the feature present but a sub-field still missing.

| Group | Reference column | No-feature rows | Feature-present-but-field-missing rows (the named exceptions) |
|---|---|---|---|
| `PoolQC`, `MiscFeature`, `Alley`, `Fence`, `FireplaceQu` | — (no exceptions in this group) | `"None"` | n/a |
| Garage (`GarageType`/`Qual`/`Finish`/`Cond`/`YrBlt`/`Cars`/`Area`) | `GarageType.notna()` | categorical → `"None"`, numeric → `0` | mode (categorical) / median (numeric), fit from rows where `GarageType` is set — covers e.g. test `Id 2577` |
| Basement (`BsmtQual`/`Cond`/`Exposure`/`FinType1`/`FinType2`/`FullBath`/`HalfBath`/`FinSF1`/`FinSF2`/`UnfSF`/`TotalBsmtSF`) | `TotalBsmtSF > 0` | categorical → `"None"`, numeric → `0` | mode / median, fit from rows where `TotalBsmtSF > 0` |
| `MasVnrType` + `MasVnrArea` | — | Type=`"None"`, Area=`0` (also covers the "neither recorded" case — the conservative default, since it's the overwhelmingly common pattern and there's no signal to say otherwise) | Area recorded but Type missing → Type imputed as the mode among veneer-present rows |

Real gaps (not structural at all):

| Column | Rule |
|---|---|
| `LotFrontage` | Median **per `Neighborhood`**, fit on train; fall back to the train-wide median for any neighborhood with no training rows to compute from. Motivated by 2_missingness's finding that the missing rate tracks `LotConfig`/neighborhood. |
| `Electrical` | Train-wide mode (`SBrkr`, ~91%). |
| `MSZoning`, `Utilities`, `Functional`, `SaleType`, `KitchenQual`, `Exterior1st`, `Exterior2nd` (test-only gaps) | Train-wide mode per column. |

`transform()` ends with `assert not X.isna().any().any()` — a cheap,
meaningful invariant, and the first thing the test suite checks.

## Component 2: `FeatureEngineer`

Operates on `Imputer`'s output — human-readable, zero-`NaN`, not yet
encoded. Every decision below cites the specific EDA finding that
justifies it.

**Structural-identity resolution** (`5_multicollinearity.ipynb`): rather
than dropping an entire total-or-parts side, drop only the one column
needed to break each identity, chosen by that notebook's own post-total-exclusion
VIF ranking — both are directly justified by numbers already computed,
not a new judgment call:
- Drop `1stFlrSF` (residual VIF 9.96). Keeps `GrLivArea` and `2ndFlrSF`
  (`LowQualFinSF` also survives this step algebraically, but its raw
  column is separately dropped below by the has-X rule — it doesn't
  end up in the final output either way).
- Drop `BsmtUnfSF` (residual VIF 9.89). Keeps `TotalBsmtSF`, `BsmtFinSF1`,
  `BsmtFinSF2`.

**Derived features**:
- `HasGarage = (GarageType != "None")`.
- `GarageAge = GarageYrBlt - YearBuilt`, set to `0` where `HasGarage` is
  `False` (its `GarageYrBlt` is the `Imputer`'s `0` sentinel, so the raw
  subtraction is meaningless there — the paired `HasGarage` flag is what
  tells the model to ignore it).

**Has-X flags** (`4_bivariate.ipynb`'s zero-inflated split):
- 5 well-supported columns get a `Has{Col}` flag **and keep** their
  magnitude (still log-transformed later, by `Encoder`): `OpenPorchSF`,
  `MasVnrArea`, `WoodDeckSF`, `2ndFlrSF`, `EnclosedPorch`.
- 3 too-sparse columns get a `Has{Col}` flag **replacing** magnitude —
  the raw column is dropped, per EDA's "too few nonzero rows to support a
  size relationship": `PoolArea`, `LowQualFinSF`, `3SsnPorch`.

**Rare-level bucketing, ordinal** (`4_bivariate.ipynb`'s 8 thin-tail
columns: `ExterCond`, `Functional`, `HeatingQC`, `BsmtCond`, `GarageCond`,
`OverallQual`, `PoolQC`, `GarageQual` — the 4 columns with *real*
inversions, `OverallCond`/`LotShape`/`LandSlope`/`BsmtFinType1`/`2`, are
explicitly left unbucketed, since EDA's finding was that their pattern
might be signal, not noise). Rule, fit on train, applied only within
these 8 columns: while any level has fewer than 10 training rows, merge
it into whichever adjacent documented-order level has more rows,
cascading until every remaining level clears the threshold (or only the
two ends are left). 10 was chosen to match the "n≤3 is noise" /
"too sparse to trust" language used throughout `3_univariate.ipynb`/`4_bivariate.ipynb`,
with margin. This changes the *category labels* only — `Encoder`'s
`ORDER` dict is untouched, since a merge only ever removes a level from
being observed, never invents one outside the documented order.

**Rare-level bucketing, nominal**: only the two columns
`4_bivariate.ipynb` explicitly reversed from "drop" to "bucket" — `RoofMatl`,
`Heating`. Non-majority levels are relabeled to a single `"Other"`
category (matches that notebook's own majority-vs-bucketed-minority
analysis directly — not the generic threshold rule above, since these
were already analyzed as a binary split, not a multi-level one).

**Column drops**: `Utilities` — near-zero-variance, undecidable (`4_bivariate.ipynb`:
minority is a single row, no comparison possible). `Street` and
`Condition2` are explicitly **not** dropped or bucketed — EDA called both
borderline/undecidable given how thin their evidence is, and this design
doesn't force a call EDA itself declined to make.

## Component 3: `Encoder` (preprocessing, last)

Operates on `FeatureEngineer`'s output. Fully mechanical — every rule
here is a direct, uncontested EDA finding:

- **Ordinal**: integer-encode all 21 columns on the documented `ORDER`
  dict (unchanged from `3_univariate.ipynb`/`4_bivariate.ipynb`).
- **Log transform**: `log1p` (handles zeros natively, no `+1` hack
  needed) on the surviving `SIZE_COLS` (`GrLivArea`, `LotArea`,
  `LotFrontage`, `TotalBsmtSF`, `BsmtFinSF1`, `GarageArea` — 6 of the
  original 8, after `FeatureEngineer` drops `1stFlrSF`/`BsmtUnfSF`) plus
  the 5 well-supported zero-inflated columns that keep their magnitude.
- **Nominal**: `OneHotEncoder(handle_unknown="ignore")`, fit on train's
  observed categories. This is also how `MSSubClass`'s test-only `150`
  gets handled — no special-case code, an unseen category just becomes an
  all-zero row. Operates on the already-bucketed nominal columns, so
  `RoofMatl`/`Heating` only ever see `"Other"` plus the majority level.

## Composition

```python
# src/ames_price/pipeline.py
def build_pipeline() -> Pipeline:
    return Pipeline([
        ("impute", Imputer()),
        ("engineer", FeatureEngineer()),
        ("encode", Encoder()),
    ])
```

`pipeline.fit_transform(df_train)` / `pipeline.transform(df_test)`. Each
step's `fit()` only ever sees what it's called with — since `Pipeline`
chains `fit_transform` on train and `transform`-only on test, there's no
path for a test-set value to influence a fitted statistic.

## Outlier handling (outside the pipeline)

Per the earlier design decision: dropping `Id` 524/1299 doesn't fit the
transform contract (rows can't vanish on the test side), so it's not a
pipeline step. Whoever calls this pipeline for training drops those two
rows from `df_train` **before** calling `fit_transform` — demonstrated in
`notebooks/preprocessing_check.ipynb` (see below), to be repeated the
same way in a future modeling script. Dropping before fitting also keeps
every fitted statistic (imputation medians, bucket thresholds) from being
influenced by them, not just the final row count.

## `notebooks/preprocessing_check.ipynb`

A lightweight verification notebook, matching the EDA series' habit of
visually confirming a transform did what it claims — not new analysis.
Loads train/test, drops the two outlier rows from train, runs
`build_pipeline()`, and checks:

- Zero `NaN` in both outputs.
- Output shape and column list (spot-check that `1stFlrSF`/`BsmtUnfSF`
  are gone, `HasGarage`/`GarageAge`/`Has{Col}` flags are present).
- Before/after histograms for 2-3 log-transformed columns, confirming the
  skew reduction already established in `1_target_variable.ipynb`
  actually happens here too.
- Value counts for a bucketed ordinal column (e.g. `ExterCond`) and a
  bucketed nominal column (`RoofMatl`), confirming the merge landed where
  expected.
- A handful of known garage/no-garage rows, spot-checking `GarageAge`
  is `0` exactly where `HasGarage` is `False`.

## Testing strategy

- **Unit tests per transformer rule**, on small synthetic `DataFrame`s —
  not the real CSV. E.g.: a row with `GarageType` set but `GarageQual`
  `NaN` gets the conditional fill, not `"None"`; a `TotalBsmtSF == 0` row
  gets `BsmtQual = "None"`; a level with 3 training rows gets merged into
  its neighbor; the surviving `1stFlrSF + 2ndFlrSF + LowQualFinSF` no
  longer sums to a retained `GrLivArea`-adjacent total once `1stFlrSF` is
  dropped (i.e. the identity is actually gone, not just relabeled).
- **One integration test** running the full pipeline on real
  `train.csv`/`test.csv` (or a fixture subset), asserting zero `NaN`,
  the expected final column set, and that fitting on train alone (no
  `y` needed anywhere) is sufficient to transform test without error.
- No test asserts on exact `SalePrice`/model performance — this is a data
  pipeline, not a model; correctness here means "matches the documented
  rule," not "produces a good score."

## Open questions (explicitly out of scope for this round)

- Final bucketing for the "encode carefully" nominal columns
  (`Neighborhood`, `Exterior1st`/`2nd`, `Condition1`, `RoofStyle`,
  `SaleType`, `Foundation`).
- `Street`/`Condition2` — left as-is per EDA's own restraint; may need
  revisiting once a model shows whether they contribute anything.
- Whether `Id` 524/1299 are dropped, downweighted, or kept with a
  `SaleCondition`-aware adjustment — this design only makes "drop before
  fit" mechanically possible, it doesn't decide it's the right call.
- Scaling/normalization for a linear model (this pipeline's output is
  log-transformed but not scaled) — deferred to whatever model consumes
  this matrix, since the right scaling depends on the model family.
- The output matrix is rank-deficient by one column per one-hot-encoded
  nominal column (`OneHotEncoder` emits a full dummy set, no `drop`,
  matching this design's own encoding rule above) — harmless for tree
  models and regularized linear models, but plain OLS will fail on it
  as-is. Whoever picks a linear model next should pass `drop="first"`
  at that point, or accept the collinearity and regularize.
