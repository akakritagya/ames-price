# Preprocessing & Feature Engineering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn raw `train.csv`/`test.csv` into a fully numeric, leak-free, model-ready matrix via three composed sklearn transformers, each rule traced to a specific EDA finding.

**Architecture:** Three `BaseEstimator`+`TransformerMixin` classes — `Imputer` (fills every NaN), `FeatureEngineer` (structural-identity resolution, derived features, rare-level bucketing), `Encoder` (ordinal/log1p/one-hot) — composed by `build_pipeline()` into one `sklearn.pipeline.Pipeline`. Outlier row-dropping stays outside the pipeline.

**Tech Stack:** Python 3.13.15, pandas, numpy, scikit-learn, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-09-04-preprocessing-feature-engineering-design.md`

## Global Constraints

- Python 3.13.15 (`pyproject.toml` `requires-python`); every command below runs via `uv run`.
- ruff: line-length 80, double quotes, rules `E,F,I,UP,B,SIM,RUF,S,PTH,C901,PLR0913` — `uv run ruff format <file>` then `uv run ruff check <file>` must be clean before each commit.
- mypy: `files = ["src"]`, `check_untyped_defs = true` — `uv run mypy src` must be clean; annotate `fit`/`transform` signatures.
- Every transformer subclasses `sklearn.base.BaseEstimator` + `sklearn.base.TransformerMixin`; `fit()` reads only the `X` it's called with; `transform()` starts with `X = X.copy()`, never mutates its input in place.
- No comments explaining *what* code does — only *why*, when non-obvious. Match the rest of the repo.
- Tests use small inline `pd.DataFrame` fixtures, not the real CSVs — except the one integration test in Task 8.

---

## Task 1: Column constants module

**Files:**
- Create: `src/ames_price/constants.py`
- Test: `tests/test_constants.py`

**Interfaces:**
- Produces: `IDENTIFIER: list[str]`, `TARGET: list[str]`, `NUMERIC_COLS: list[str]`, `ORDINAL_COLS: list[str]`, `NOMINAL_COLS: list[str]`, `ORDER: dict[str, list]` — every later task imports from here instead of redefining these lists.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_constants.py
from ames_price.constants import (
    IDENTIFIER,
    NOMINAL_COLS,
    NUMERIC_COLS,
    ORDER,
    ORDINAL_COLS,
    TARGET,
)


def test_column_groups_are_disjoint_and_cover_expected_counts():
    assert len(NUMERIC_COLS) == 33
    assert len(ORDINAL_COLS) == 21
    assert len(NOMINAL_COLS) == 25
    all_cols = NUMERIC_COLS + ORDINAL_COLS + NOMINAL_COLS
    assert len(all_cols) == len(set(all_cols))


def test_order_covers_exactly_the_ordinal_columns():
    assert set(ORDER.keys()) == set(ORDINAL_COLS)


def test_identifier_and_target_are_singletons():
    assert IDENTIFIER == ["Id"]
    assert TARGET == ["SalePrice"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_constants.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ames_price.constants'`

- [ ] **Step 3: Write the implementation**

```python
# src/ames_price/constants.py
"""Column groupings and documented category orders, from data_description.txt.

Single source of truth for the feature-type split used by both the EDA
notebooks and the preprocessing/feature-engineering pipeline.
"""

IDENTIFIER = ["Id"]
TARGET = ["SalePrice"]

NUMERIC_COLS = sorted(
    [
        "LotFrontage",
        "LotArea",
        "MasVnrArea",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "1stFlrSF",
        "2ndFlrSF",
        "LowQualFinSF",
        "GrLivArea",
        "BsmtFullBath",
        "BsmtHalfBath",
        "FullBath",
        "HalfBath",
        "BedroomAbvGr",
        "KitchenAbvGr",
        "TotRmsAbvGrd",
        "Fireplaces",
        "GarageCars",
        "GarageArea",
        "WoodDeckSF",
        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch",
        "PoolArea",
        "MiscVal",
        "YearBuilt",
        "YearRemodAdd",
        "GarageYrBlt",
        "MoSold",
        "YrSold",
    ]
)

ORDINAL_COLS = sorted(
    [
        "OverallQual",
        "OverallCond",
        "LotShape",
        "LandSlope",
        "ExterQual",
        "ExterCond",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "HeatingQC",
        "KitchenQual",
        "Functional",
        "FireplaceQu",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "PavedDrive",
        "PoolQC",
        "Utilities",
    ]
)

NOMINAL_COLS = sorted(
    [
        "MSSubClass",
        "MSZoning",
        "Street",
        "Alley",
        "LandContour",
        "LotConfig",
        "Neighborhood",
        "Condition1",
        "Condition2",
        "BldgType",
        "HouseStyle",
        "RoofStyle",
        "RoofMatl",
        "Exterior1st",
        "Exterior2nd",
        "MasVnrType",
        "Foundation",
        "Heating",
        "CentralAir",
        "Electrical",
        "GarageType",
        "MiscFeature",
        "SaleType",
        "SaleCondition",
        "Fence",
    ]
)

ORDER = {
    "OverallQual": list(range(1, 11)),
    "OverallCond": list(range(1, 11)),
    "LotShape": ["IR3", "IR2", "IR1", "Reg"],
    "LandSlope": ["Sev", "Mod", "Gtl"],
    "ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtCond": ["Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtExposure": ["No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
    "KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    "FireplaceQu": ["Po", "Fa", "TA", "Gd", "Ex"],
    "GarageFinish": ["Unf", "RFn", "Fin"],
    "GarageQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "GarageCond": ["Po", "Fa", "TA", "Gd", "Ex"],
    "PavedDrive": ["N", "P", "Y"],
    "PoolQC": ["Fa", "TA", "Gd", "Ex"],
    "Utilities": ["ELO", "NoSeWa", "NoSewr", "AllPub"],
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_constants.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Lint, format, typecheck**

Run: `uv run ruff format src/ames_price/constants.py tests/test_constants.py && uv run ruff check src/ames_price/constants.py tests/test_constants.py && uv run mypy src`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/ames_price/constants.py tests/test_constants.py
git commit -m "feat: add shared column-grouping constants module"
```

---

## Task 2: `Imputer` — structural-absence groups

**Files:**
- Create: `src/ames_price/preprocessing.py`
- Test: `tests/test_preprocessing.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `class Imputer(BaseEstimator, TransformerMixin)` with `fit(self, X: pd.DataFrame, y=None) -> "Imputer"` and `transform(self, X: pd.DataFrame) -> pd.DataFrame`. Task 3 extends this same class in this same file.

This task's fixture (`_imputer_fixture()`) covers every column `Imputer` will ever touch, including the real-gap columns Task 3 handles — Task 2's tests just don't assert on those yet. Reusing one fixture across both tasks keeps Task 3 from needing a second, inconsistent fixture.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_preprocessing.py
import numpy as np
import pandas as pd

from ames_price.preprocessing import Imputer


def _imputer_fixture() -> pd.DataFrame:
    # row 0: nothing present (no garage/basement/veneer/simple features),
    #        second Somerst-neighborhood real-gap row
    # row 1: everything present -- the baseline used to fit conditional
    #        stats for the "feature present, one sub-field missing" rows
    # row 2: the named-exception case -- garage/basement/veneer genuinely
    #        present, but a sub-field is still NaN
    # row 3: second no-feature row, in Somerst, for per-neighborhood
    #        median testing
    return pd.DataFrame(
        {
            "Neighborhood": ["CollgCr", "CollgCr", "Somerst", "Somerst"],
            "LotFrontage": [70.0, np.nan, 100.0, np.nan],
            "MSZoning": ["RL", "RL", "RL", np.nan],
            "Utilities": ["AllPub", "AllPub", "AllPub", "AllPub"],
            "Functional": ["Typ", "Typ", "Typ", "Typ"],
            "SaleType": ["WD", "WD", "WD", "WD"],
            "KitchenQual": ["TA", "TA", "TA", "TA"],
            "Exterior1st": ["VinylSd", "VinylSd", "VinylSd", "VinylSd"],
            "Exterior2nd": ["VinylSd", "VinylSd", "VinylSd", "VinylSd"],
            "Electrical": ["SBrkr", "SBrkr", "SBrkr", "SBrkr"],
            "PoolQC": [np.nan, "Gd", np.nan, np.nan],
            "MiscFeature": [np.nan, "Shed", np.nan, np.nan],
            "Alley": [np.nan, "Grvl", np.nan, np.nan],
            "Fence": [np.nan, "MnPrv", np.nan, np.nan],
            "FireplaceQu": [np.nan, "TA", np.nan, np.nan],
            "GarageType": [np.nan, "Attchd", "Attchd", np.nan],
            "GarageQual": [np.nan, "TA", np.nan, np.nan],
            "GarageFinish": [np.nan, "RFn", np.nan, np.nan],
            "GarageCond": [np.nan, "TA", np.nan, np.nan],
            "GarageYrBlt": [np.nan, 2005.0, np.nan, np.nan],
            "GarageCars": [np.nan, 2.0, np.nan, np.nan],
            "GarageArea": [np.nan, 500.0, np.nan, np.nan],
            "TotalBsmtSF": [np.nan, 800.0, 800.0, np.nan],
            "BsmtQual": [np.nan, "TA", np.nan, np.nan],
            "BsmtCond": [np.nan, "TA", np.nan, np.nan],
            "BsmtExposure": [np.nan, "No", np.nan, np.nan],
            "BsmtFinType1": [np.nan, "GLQ", np.nan, np.nan],
            "BsmtFinType2": [np.nan, "Unf", np.nan, np.nan],
            "BsmtFinSF1": [np.nan, 600.0, 600.0, np.nan],
            "BsmtFinSF2": [np.nan, 0.0, 0.0, np.nan],
            "BsmtUnfSF": [np.nan, 200.0, 200.0, np.nan],
            "BsmtFullBath": [np.nan, 1.0, 1.0, np.nan],
            "BsmtHalfBath": [np.nan, 0.0, 0.0, np.nan],
            "MasVnrType": [np.nan, "BrkFace", np.nan, np.nan],
            "MasVnrArea": [0.0, 150.0, 150.0, 0.0],
        }
    )


def test_simple_structural_groups_fill_with_none():
    result = Imputer().fit_transform(_imputer_fixture())
    for col in ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"]:
        assert result.loc[0, col] == "None"
        assert result.loc[1, col] != "None"


def test_no_garage_row_gets_none_and_zero_fill():
    result = Imputer().fit_transform(_imputer_fixture())
    assert result.loc[0, "GarageType"] == "None"
    assert result.loc[0, "GarageQual"] == "None"
    assert result.loc[0, "GarageYrBlt"] == 0
    assert result.loc[0, "GarageCars"] == 0


def test_real_garage_missing_subfield_gets_conditional_fill():
    result = Imputer().fit_transform(_imputer_fixture())
    assert result.loc[2, "GarageQual"] == "TA"
    assert result.loc[2, "GarageFinish"] == "RFn"
    assert result.loc[2, "GarageYrBlt"] == 2005.0
    assert result.loc[2, "GarageCars"] == 2.0


def test_no_basement_row_gets_none_and_zero_fill():
    result = Imputer().fit_transform(_imputer_fixture())
    assert result.loc[0, "TotalBsmtSF"] == 0
    assert result.loc[0, "BsmtQual"] == "None"
    assert result.loc[0, "BsmtFinSF1"] == 0


def test_real_basement_missing_subfield_gets_conditional_fill():
    result = Imputer().fit_transform(_imputer_fixture())
    assert result.loc[2, "BsmtQual"] == "TA"
    assert result.loc[2, "BsmtExposure"] == "No"


def test_no_veneer_gets_none_type():
    result = Imputer().fit_transform(_imputer_fixture())
    assert result.loc[0, "MasVnrType"] == "None"


def test_veneer_area_recorded_type_missing_gets_conditional_fill():
    result = Imputer().fit_transform(_imputer_fixture())
    assert result.loc[2, "MasVnrType"] == "BrkFace"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_preprocessing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ames_price.preprocessing'`

- [ ] **Step 3: Write the implementation**

```python
# src/ames_price/preprocessing.py
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

_SIMPLE_STRUCTURAL_COLS = [
    "PoolQC",
    "MiscFeature",
    "Alley",
    "Fence",
    "FireplaceQu",
]
_GARAGE_CATEGORICAL = ["GarageQual", "GarageFinish", "GarageCond"]
_GARAGE_NUMERIC = ["GarageYrBlt", "GarageCars", "GarageArea"]
_BSMT_CATEGORICAL = [
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
]
_BSMT_NUMERIC = [
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
]


class Imputer(BaseEstimator, TransformerMixin):
    """Fills every NaN using statistics fit on training data only.

    Structural-absence NaNs (no pool/garage/basement/veneer) get a fixed
    "None"/0 fill. A handful of rows have the feature present but one
    sub-field still missing (e.g. a real garage with GarageQual
    unrecorded) -- those get a statistic fit from other rows that do
    have the feature, not the structural default.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> "Imputer":
        has_garage = X["GarageType"].notna()
        self.garage_categorical_mode_ = (
            X.loc[has_garage, _GARAGE_CATEGORICAL].mode().iloc[0]
        )
        self.garage_numeric_median_ = X.loc[
            has_garage, _GARAGE_NUMERIC
        ].median()

        has_bsmt = X["TotalBsmtSF"].fillna(0) > 0
        self.bsmt_categorical_mode_ = (
            X.loc[has_bsmt, _BSMT_CATEGORICAL].mode().iloc[0]
        )

        has_veneer = X["MasVnrArea"].fillna(0) > 0
        self.masvnr_type_mode_ = X.loc[has_veneer, "MasVnrType"].mode().iloc[0]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for col in _SIMPLE_STRUCTURAL_COLS:
            X[col] = X[col].fillna("None")

        has_garage = X["GarageType"].notna()
        no_garage = ~has_garage
        for col in _GARAGE_CATEGORICAL:
            X.loc[no_garage, col] = X.loc[no_garage, col].fillna("None")
            X.loc[has_garage, col] = X.loc[has_garage, col].fillna(
                self.garage_categorical_mode_[col]
            )
        for col in _GARAGE_NUMERIC:
            X.loc[no_garage, col] = X.loc[no_garage, col].fillna(0)
            X.loc[has_garage, col] = X.loc[has_garage, col].fillna(
                self.garage_numeric_median_[col]
            )
        X["GarageType"] = X["GarageType"].fillna("None")

        has_bsmt = X["TotalBsmtSF"].fillna(0) > 0
        no_bsmt = ~has_bsmt
        for col in _BSMT_CATEGORICAL:
            X.loc[no_bsmt, col] = X.loc[no_bsmt, col].fillna("None")
            X.loc[has_bsmt, col] = X.loc[has_bsmt, col].fillna(
                self.bsmt_categorical_mode_[col]
            )
        for col in _BSMT_NUMERIC:
            X[col] = X[col].fillna(0)

        has_veneer = X["MasVnrArea"].fillna(0) > 0
        X.loc[~has_veneer, "MasVnrType"] = X.loc[
            ~has_veneer, "MasVnrType"
        ].fillna("None")
        X.loc[has_veneer, "MasVnrType"] = X.loc[
            has_veneer, "MasVnrType"
        ].fillna(self.masvnr_type_mode_)
        X["MasVnrArea"] = X["MasVnrArea"].fillna(0)

        return X
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_preprocessing.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Lint, format, typecheck**

Run: `uv run ruff format src/ames_price/preprocessing.py tests/test_preprocessing.py && uv run ruff check src/ames_price/preprocessing.py tests/test_preprocessing.py && uv run mypy src`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/ames_price/preprocessing.py tests/test_preprocessing.py
git commit -m "feat: add Imputer structural-absence fills"
```

---

## Task 3: `Imputer` — real gaps + zero-NaN invariant

**Files:**
- Modify: `src/ames_price/preprocessing.py`
- Modify: `tests/test_preprocessing.py`

**Interfaces:**
- Consumes: `Imputer` from Task 2 (same class, same file — extends `fit`/`transform`).
- Produces: `Imputer.transform()` now guarantees zero remaining `NaN` — Task 8's integration test relies on this.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_preprocessing.py`:

```python
def test_lotfrontage_filled_with_neighborhood_median():
    result = Imputer().fit_transform(_imputer_fixture())
    assert result.loc[1, "LotFrontage"] == 70.0
    assert result.loc[3, "LotFrontage"] == 100.0


def test_mszoning_filled_with_mode():
    result = Imputer().fit_transform(_imputer_fixture())
    assert result.loc[3, "MSZoning"] == "RL"


def test_transform_leaves_no_nan():
    result = Imputer().fit_transform(_imputer_fixture())
    assert not result.isna().any().any()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_preprocessing.py -v`
Expected: `test_lotfrontage_filled_with_neighborhood_median` and `test_mszoning_filled_with_mode` FAIL (columns still `NaN`); `test_transform_leaves_no_nan` FAILs too.

- [ ] **Step 3: Extend the implementation**

In `src/ames_price/preprocessing.py`, add below the existing group constants:

```python
_REAL_GAP_MODE_COLS = [
    "MSZoning",
    "Utilities",
    "Functional",
    "SaleType",
    "KitchenQual",
    "Exterior1st",
    "Exterior2nd",
    "Electrical",
]
```

In `Imputer.fit`, add before `return self`:

```python
self.lotfrontage_by_neighborhood_ = X.groupby("Neighborhood")[
    "LotFrontage"
].median()
self.lotfrontage_global_median_ = X["LotFrontage"].median()
self.mode_fills_ = {col: X[col].mode().iloc[0] for col in _REAL_GAP_MODE_COLS}
```

In `Imputer.transform`, replace the final `return X` with:

```python
neighborhood_fill = X["Neighborhood"].map(self.lotfrontage_by_neighborhood_)
X["LotFrontage"] = (
    X["LotFrontage"]
    .fillna(neighborhood_fill)
    .fillna(self.lotfrontage_global_median_)
)

for col, fill in self.mode_fills_.items():
    X[col] = X[col].fillna(fill)

assert not X.isna().any().any(), (
    f"Imputer left NaNs in: {X.columns[X.isna().any()].tolist()}"
)
return X
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_preprocessing.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Lint, format, typecheck**

Run: `uv run ruff format src/ames_price/preprocessing.py tests/test_preprocessing.py && uv run ruff check src/ames_price/preprocessing.py tests/test_preprocessing.py && uv run mypy src`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/ames_price/preprocessing.py tests/test_preprocessing.py
git commit -m "feat: complete Imputer with real-gap fills and zero-NaN invariant"
```

---

## Task 4: `FeatureEngineer` — identity resolution, derived features, has-X flags

**Files:**
- Create: `src/ames_price/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (operates on `Imputer`'s output shape, but doesn't import it).
- Produces: `class FeatureEngineer(BaseEstimator, TransformerMixin)` with `fit`/`transform`. Task 5 extends this same class in this same file.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_features.py
import numpy as np
import pandas as pd

from ames_price.features import FeatureEngineer


def _feature_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "1stFlrSF": [800.0, 1200.0],
            "2ndFlrSF": [0.0, 600.0],
            "LowQualFinSF": [0.0, 0.0],
            "GrLivArea": [800.0, 1800.0],
            "BsmtFinSF1": [400.0, 0.0],
            "BsmtFinSF2": [0.0, 0.0],
            "BsmtUnfSF": [400.0, 900.0],
            "TotalBsmtSF": [800.0, 900.0],
            "GarageType": ["None", "Attchd"],
            "GarageYrBlt": [0.0, 2005.0],
            "YearBuilt": [1990.0, 2003.0],
            "OpenPorchSF": [0.0, 45.0],
            "MasVnrArea": [0.0, 200.0],
            "WoodDeckSF": [0.0, 100.0],
            "EnclosedPorch": [0.0, 0.0],
            "PoolArea": [0.0, 0.0],
            "3SsnPorch": [0.0, 0.0],
        }
    )


def test_structural_identity_columns_are_dropped():
    result = FeatureEngineer().fit_transform(_feature_fixture())
    assert "1stFlrSF" not in result.columns
    assert "BsmtUnfSF" not in result.columns
    assert "GrLivArea" in result.columns
    assert "TotalBsmtSF" in result.columns


def test_has_garage_and_garage_age():
    result = FeatureEngineer().fit_transform(_feature_fixture())
    assert result.loc[0, "HasGarage"] == False  # noqa: E712
    assert result.loc[0, "GarageAge"] == 0
    assert result.loc[1, "HasGarage"] == True  # noqa: E712
    assert result.loc[1, "GarageAge"] == 2005.0 - 2003.0


def test_has_x_flags_keep_magnitude():
    result = FeatureEngineer().fit_transform(_feature_fixture())
    assert result.loc[0, "HasOpenPorchSF"] == False  # noqa: E712
    assert result.loc[1, "HasOpenPorchSF"] == True  # noqa: E712
    assert result.loc[1, "OpenPorchSF"] == 45.0


def test_has_x_flags_drop_magnitude():
    result = FeatureEngineer().fit_transform(_feature_fixture())
    assert "PoolArea" not in result.columns
    assert "HasPoolArea" in result.columns
    assert result.loc[0, "HasPoolArea"] == False  # noqa: E712
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ames_price.features'`

- [ ] **Step 3: Write the implementation**

```python
# src/ames_price/features.py
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

_HAS_KEEP_MAGNITUDE_COLS = [
    "OpenPorchSF",
    "MasVnrArea",
    "WoodDeckSF",
    "2ndFlrSF",
    "EnclosedPorch",
]
_HAS_DROP_MAGNITUDE_COLS = ["PoolArea", "LowQualFinSF", "3SsnPorch"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Structural-identity resolution, derived features, has-X flags, and
    rare-level bucketing -- the judgment calls EDA flagged but left open.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> "FeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        X = X.drop(columns=["1stFlrSF", "BsmtUnfSF"])

        X["HasGarage"] = X["GarageType"] != "None"
        X["GarageAge"] = np.where(
            X["HasGarage"], X["GarageYrBlt"] - X["YearBuilt"], 0
        )

        for col in _HAS_KEEP_MAGNITUDE_COLS:
            X[f"Has{col}"] = X[col] > 0

        for col in _HAS_DROP_MAGNITUDE_COLS:
            X[f"Has{col}"] = X[col] > 0
            X = X.drop(columns=[col])

        return X
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_features.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Lint, format, typecheck**

Run: `uv run ruff format src/ames_price/features.py tests/test_features.py && uv run ruff check src/ames_price/features.py tests/test_features.py && uv run mypy src`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/ames_price/features.py tests/test_features.py
git commit -m "feat: add FeatureEngineer identity resolution and has-X flags"
```

---

## Task 5: `FeatureEngineer` — rare-level bucketing + `Utilities` drop

**Files:**
- Modify: `src/ames_price/features.py`
- Modify: `tests/test_features.py`

**Interfaces:**
- Consumes: `FeatureEngineer` from Task 4 (same class, same file); `ORDER` from `ames_price.constants` (Task 1).
- Produces: `FeatureEngineer.transform()` now also bucketed rare ordinal/nominal levels and drops `Utilities` — Task 8's integration test relies on the resulting column set.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_features.py`:

```python
def _bucket_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ExterCond": ["Po", *["TA"] * 12, "Ex"],
            "RoofMatl": [*["CompShg"] * 12, "WdShngl", "WdShake"],
            "Heating": [*["GasA"] * 12, "GasW", "Wall"],
            "Utilities": ["AllPub"] * 14,
        }
    )


def test_ordinal_thin_levels_merge_into_majority():
    result = FeatureEngineer().fit_transform(_bucket_fixture())
    assert result.loc[0, "ExterCond"] == "TA"
    assert result.loc[13, "ExterCond"] == "TA"
    assert (result["ExterCond"] == "TA").all()


def test_nominal_minority_buckets_into_other():
    result = FeatureEngineer().fit_transform(_bucket_fixture())
    assert result.loc[12, "RoofMatl"] == "Other"
    assert result.loc[13, "RoofMatl"] == "Other"
    assert (result.loc[:11, "RoofMatl"] == "CompShg").all()


def test_utilities_is_dropped():
    result = FeatureEngineer().fit_transform(_bucket_fixture())
    assert "Utilities" not in result.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_features.py -v`
Expected: `test_ordinal_thin_levels_merge_into_majority`, `test_nominal_minority_buckets_into_other`, `test_utilities_is_dropped` FAIL (`Utilities` still present, `ExterCond`/`RoofMatl` unchanged).

- [ ] **Step 3: Extend the implementation**

In `src/ames_price/features.py`, add the import and new constants above the class:

```python
from ames_price.constants import ORDER

_ORDINAL_BUCKET_COLS = [
    "ExterCond",
    "Functional",
    "HeatingQC",
    "BsmtCond",
    "GarageCond",
    "OverallQual",
    "PoolQC",
    "GarageQual",
]
_NOMINAL_OTHER_BUCKET_COLS = ["RoofMatl", "Heating"]
_MIN_LEVEL_COUNT = 10


def _build_merge_map(series: pd.Series, order: list, min_count: int) -> dict:
    """Cascading merge: repeatedly find the globally thinnest level and
    merge it into whichever adjacent documented-order level is more
    populous, until every remaining group clears min_count (or only one
    group is left). Keeps the survivor's own label, never invents a new
    one -- Encoder's ORDER dict stays valid unchanged."""
    counts = [int((series == level).sum()) for level in order]
    members = [[level] for level in order]
    reps = list(order)

    while len(members) > 1 and min(counts) < min_count:
        i = counts.index(min(counts))
        if i == 0:
            j = 1
        elif i == len(members) - 1:
            j = i - 1
        else:
            j = i - 1 if counts[i - 1] >= counts[i + 1] else i + 1

        keep, drop = min(i, j), max(i, j)
        members[keep] = (
            members[i] + members[j] if i < j else members[j] + members[i]
        )
        counts[keep] = counts[i] + counts[j]
        reps[keep] = reps[j]

        del members[drop]
        del counts[drop]
        del reps[drop]

    return {level: rep for group, rep in zip(members, reps) for level in group}
```

In `FeatureEngineer.fit`, replace `return self` with:

```python
        self.ordinal_merge_maps_ = {
            col: _build_merge_map(X[col], ORDER[col], _MIN_LEVEL_COUNT)
            for col in _ORDINAL_BUCKET_COLS
        }
        self.nominal_majority_ = {
            col: X[col].mode().iloc[0] for col in _NOMINAL_OTHER_BUCKET_COLS
        }
        return self
```

In `FeatureEngineer.transform`, replace the final `return X` with:

```python
        for col, merge_map in self.ordinal_merge_maps_.items():
            X[col] = X[col].map(merge_map)

        for col in _NOMINAL_OTHER_BUCKET_COLS:
            majority = self.nominal_majority_[col]
            X[col] = np.where(X[col] == majority, majority, "Other")

        X = X.drop(columns=["Utilities"])

        return X
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_features.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Lint, format, typecheck**

Run: `uv run ruff format src/ames_price/features.py tests/test_features.py && uv run ruff check src/ames_price/features.py tests/test_features.py && uv run mypy src`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/ames_price/features.py tests/test_features.py
git commit -m "feat: add FeatureEngineer rare-level bucketing and Utilities drop"
```

---

## Task 6: `Encoder` — ordinal encoding + log1p transform

**Files:**
- Modify: `src/ames_price/preprocessing.py`
- Modify: `tests/test_preprocessing.py`

**Interfaces:**
- Consumes: `ORDER`, `ORDINAL_COLS` from `ames_price.constants` (Task 1).
- Produces: `class Encoder(BaseEstimator, TransformerMixin)` in the same file as `Imputer`. Task 7 extends this same class.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_preprocessing.py`:

```python
from ames_price.preprocessing import Encoder


def test_ordinal_columns_encoded_on_documented_order():
    X = pd.DataFrame({"KitchenQual": ["Po", "TA", "Ex"]})
    result = Encoder().fit_transform(X)
    assert result["KitchenQual"].tolist() == [0, 2, 4]


def test_log1p_applied_to_flagged_numeric_columns():
    X = pd.DataFrame({"GrLivArea": [0.0, 999.0]})
    result = Encoder().fit_transform(X)
    assert result["GrLivArea"].tolist() == [np.log1p(0.0), np.log1p(999.0)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_preprocessing.py -v`
Expected: FAIL with `ImportError: cannot import name 'Encoder'`

- [ ] **Step 3: Write the implementation**

Append to `src/ames_price/preprocessing.py` (add `import numpy as np` and `from ames_price.constants import ORDER, ORDINAL_COLS` at the top of the file alongside the existing imports):

```python
_LOG1P_COLS = [
    "GrLivArea",
    "LotArea",
    "LotFrontage",
    "TotalBsmtSF",
    "BsmtFinSF1",
    "GarageArea",
    "OpenPorchSF",
    "MasVnrArea",
    "WoodDeckSF",
    "2ndFlrSF",
    "EnclosedPorch",
]


class Encoder(BaseEstimator, TransformerMixin):
    """Ordinal integer-encoding on the documented order, log1p on skewed
    numerics, one-hot on nominal categories. Fully mechanical -- every
    rule here is a direct EDA finding, nothing decided in this class.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> "Encoder":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for col in ORDINAL_COLS:
            if col in X.columns:
                X[col] = X[col].map(
                    {level: i for i, level in enumerate(ORDER[col])}
                )

        for col in _LOG1P_COLS:
            if col in X.columns:
                X[col] = np.log1p(X[col])

        return X
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_preprocessing.py -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Lint, format, typecheck**

Run: `uv run ruff format src/ames_price/preprocessing.py tests/test_preprocessing.py && uv run ruff check src/ames_price/preprocessing.py tests/test_preprocessing.py && uv run mypy src`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/ames_price/preprocessing.py tests/test_preprocessing.py
git commit -m "feat: add Encoder ordinal encoding and log1p transform"
```

---

## Task 7: `Encoder` — nominal one-hot encoding

**Files:**
- Modify: `src/ames_price/preprocessing.py`
- Modify: `tests/test_preprocessing.py`

**Interfaces:**
- Consumes: `NOMINAL_COLS` from `ames_price.constants` (Task 1); `sklearn.preprocessing.OneHotEncoder`.
- Produces: `Encoder.transform()` now returns a fully numeric matrix (ordinal + log1p + one-hot) — Task 8's integration test relies on this being the final shape.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_preprocessing.py`:

```python
def test_nominal_columns_are_one_hot_encoded():
    X_train = pd.DataFrame({"Street": ["Pave", "Pave", "Grvl"]})
    encoder = Encoder().fit(X_train)
    result = encoder.transform(X_train)
    assert "Street" not in result.columns
    assert "Street_Pave" in result.columns
    assert "Street_Grvl" in result.columns
    assert result.loc[0, "Street_Pave"] == 1.0
    assert result.loc[2, "Street_Grvl"] == 1.0


def test_unseen_category_at_transform_time_becomes_all_zero_row():
    X_train = pd.DataFrame({"MSSubClass": [20, 60, 20]})
    X_test = pd.DataFrame({"MSSubClass": [20, 150]})
    encoder = Encoder().fit(X_train)
    result = encoder.transform(X_test)
    assert result.loc[1, "MSSubClass_20"] == 0.0
    assert result.loc[1, "MSSubClass_60"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_preprocessing.py -v`
Expected: `test_nominal_columns_are_one_hot_encoded` and `test_unseen_category_at_transform_time_becomes_all_zero_row` FAIL (`Street`/`MSSubClass` still present unchanged, no `_Pave`/`_20` columns).

- [ ] **Step 3: Extend the implementation**

In `src/ames_price/preprocessing.py`, add to the imports:

```python
from sklearn.preprocessing import OneHotEncoder

from ames_price.constants import NOMINAL_COLS, ORDER, ORDINAL_COLS
```

In `Encoder.fit`, replace `return self` with:

```python
        self.nominal_cols_ = [c for c in NOMINAL_COLS if c in X.columns]
        self.onehot_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.onehot_.fit(X[self.nominal_cols_])
        return self
```

In `Encoder.transform`, replace the final `return X` with:

```python
        onehot = self.onehot_.transform(X[self.nominal_cols_])
        onehot_df = pd.DataFrame(
            onehot,
            columns=self.onehot_.get_feature_names_out(self.nominal_cols_),
            index=X.index,
        )
        X = X.drop(columns=self.nominal_cols_)
        return pd.concat([X, onehot_df], axis=1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_preprocessing.py -v`
Expected: PASS (14 tests)

- [ ] **Step 5: Lint, format, typecheck**

Run: `uv run ruff format src/ames_price/preprocessing.py tests/test_preprocessing.py && uv run ruff check src/ames_price/preprocessing.py tests/test_preprocessing.py && uv run mypy src`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/ames_price/preprocessing.py tests/test_preprocessing.py
git commit -m "feat: add Encoder nominal one-hot encoding"
```

---

## Task 8: `build_pipeline()` + integration test

**Files:**
- Create: `src/ames_price/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `Imputer`, `Encoder` from `ames_price.preprocessing` (Tasks 2/3/6/7); `FeatureEngineer` from `ames_price.features` (Tasks 4/5); `NUMERIC_COLS`, `ORDINAL_COLS`, `NOMINAL_COLS` from `ames_price.constants` (Task 1).
- Produces: `build_pipeline() -> sklearn.pipeline.Pipeline`.

- [ ] **Step 1: Write the failing test**

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
    assert list(train_out.columns) == list(test_out.columns)
    assert all(np.issubdtype(dtype, np.number) for dtype in train_out.dtypes)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ames_price.pipeline'`

- [ ] **Step 3: Write the implementation**

```python
# src/ames_price/pipeline.py
from __future__ import annotations

from sklearn.pipeline import Pipeline

from ames_price.features import FeatureEngineer
from ames_price.preprocessing import Encoder, Imputer


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("impute", Imputer()),
            ("engineer", FeatureEngineer()),
            ("encode", Encoder()),
        ]
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest -v`
Expected: all tests across `test_constants.py`, `test_preprocessing.py`, `test_features.py`, `test_pipeline.py`, and the pre-existing `test_main.py` PASS.

- [ ] **Step 6: Lint, format, typecheck**

Run: `uv run ruff format src/ames_price/pipeline.py tests/test_pipeline.py && uv run ruff check src/ames_price/pipeline.py tests/test_pipeline.py && uv run mypy src`
Expected: clean

- [ ] **Step 7: Commit**

```bash
git add src/ames_price/pipeline.py tests/test_pipeline.py
git commit -m "feat: compose Imputer, FeatureEngineer, Encoder into build_pipeline()"
```

---

## Task 9: `notebooks/preprocessing_check.ipynb`

**Files:**
- Create: `notebooks/preprocessing_check.ipynb`
- Create (temporary, scratchpad): a small Python script that builds the notebook JSON, matching the pattern used for `notebooks/eda/*.ipynb`

**Interfaces:**
- Consumes: `build_pipeline()` from `ames_price.pipeline` (Task 8); `NUMERIC_COLS`/`ORDINAL_COLS`/`NOMINAL_COLS` from `ames_price.constants` (Task 1).
- Produces: an executed, committed notebook — no code elsewhere depends on it.

- [ ] **Step 1: Write the notebook-build script**

```python
# scratchpad script (not committed) -- builds notebooks/preprocessing_check.ipynb
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
        "# Preprocessing check\n\n"
        "Runs `build_pipeline()` end to end and visually confirms it did "
        "what `docs/superpowers/specs/2026-09-04-preprocessing-feature-engineering-design.md` "
        "claims -- not new analysis, a verification pass, matching the "
        "EDA series' habit of checking a transform rather than trusting it."
    ),
    code(
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n\n"
        "from ames_price.constants import NOMINAL_COLS, NUMERIC_COLS, ORDINAL_COLS\n"
        "from ames_price.pipeline import build_pipeline\n\n"
        "FEATURE_COLS = NUMERIC_COLS + ORDINAL_COLS + NOMINAL_COLS"
    ),
    code(
        'df_train = pd.read_csv("../data/train.csv")\n'
        'df_test = pd.read_csv("../data/test.csv")\n\n'
        "# per the design doc: drop the two flagged outliers before fitting\n"
        "# anything, so no imputation/bucketing statistic is influenced by them\n"
        'df_train = df_train[~df_train["Id"].isin([524, 1299])]\n'
        "df_train.shape, df_test.shape"
    ),
    code(
        "pipeline = build_pipeline()\n"
        "train_out = pipeline.fit_transform(df_train[FEATURE_COLS])\n"
        "test_out = pipeline.transform(df_test[FEATURE_COLS])\n\n"
        "assert not train_out.isna().any().any()\n"
        "assert not test_out.isna().any().any()\n"
        "train_out.shape, test_out.shape"
    ),
    md(
        "## Column-set sanity check\n\nConfirm the identity columns are gone and the new derived/flag columns are present."
    ),
    code(
        'for col in ["1stFlrSF", "BsmtUnfSF", "Utilities"]:\n'
        "    assert col not in train_out.columns, col\n"
        'for col in ["HasGarage", "GarageAge", "HasOpenPorchSF", "HasPoolArea"]:\n'
        "    assert col in train_out.columns, col\n"
        'print("ok")'
    ),
    md(
        "## Log-transform check\n\nBefore/after histograms for a couple of the flagged skewed columns -- confirming the skew reduction shows up here, not just in `1_target_variable.ipynb`."
    ),
    code(
        "fig, axes = plt.subplots(2, 2, figsize=(10, 7))\n"
        'sns_cols = ["GrLivArea", "LotArea"]\n'
        "for i, col in enumerate(sns_cols):\n"
        "    axes[0, i].hist(df_train[col].dropna(), bins=40)\n"
        '    axes[0, i].set_title(f"{col} (raw)")\n'
        "    axes[1, i].hist(train_out[col], bins=40)\n"
        '    axes[1, i].set_title(f"{col} (log1p)")\n'
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "## Bucketing check\n\nValue counts for a bucketed ordinal column and a bucketed nominal column, confirming the merge landed where expected."
    ),
    code(
        'print("ExterCond (post-Encoder, integer-coded):")\n'
        'print(train_out["ExterCond"].value_counts().sort_index())'
    ),
    code(
        'roofmatl_cols = [c for c in train_out.columns if c.startswith("RoofMatl_")]\n'
        "print(train_out[roofmatl_cols].sum())"
    ),
    md(
        "## Derived-feature spot check\n\n`GarageAge` should be exactly 0 wherever `HasGarage` is `False`."
    ),
    code(
        'no_garage = ~train_out["HasGarage"]\n'
        'assert (train_out.loc[no_garage, "GarageAge"] == 0).all()\n'
        'print(f"{no_garage.sum()} no-garage rows, all GarageAge == 0")'
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
with open("notebooks/preprocessing_check.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
```

- [ ] **Step 2: Run the script and execute the notebook**

Run:
```bash
python3 /path/to/scratchpad/build_preprocessing_check_nb.py
cd notebooks && uv run --project .. jupyter nbconvert --to notebook --execute --inplace preprocessing_check.ipynb
```
Expected: `[NbConvertApp] Writing ... bytes to preprocessing_check.ipynb`, no error cells.

- [ ] **Step 3: Verify no errors**

Run:
```python
import json

nb = json.load(open("notebooks/preprocessing_check.ipynb"))
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

- [ ] **Step 4: Commit**

```bash
git add notebooks/preprocessing_check.ipynb
git commit -m "feat: add preprocessing_check notebook verifying the pipeline output"
```

---

## Self-Review Notes

- **Spec coverage:** `Imputer` (Tasks 2-3), `FeatureEngineer` (Tasks 4-5), `Encoder` (Tasks 6-7), `build_pipeline()` (Task 8), `notebooks/preprocessing_check.ipynb` (Task 9) — every component in the spec has a task. Outlier row-dropping is demonstrated in Task 9's notebook per the spec's "outside the pipeline" decision, not implemented as pipeline code (correctly — the spec says it's the caller's job).
- **Type consistency checked:** `Imputer`/`FeatureEngineer`/`Encoder` all return `pd.DataFrame` from `transform`; `build_pipeline()`'s step names (`"impute"`, `"engineer"`, `"encode"`) match no other task's assumptions (none reference step names directly). Column names introduced in Task 4/5 (`HasGarage`, `GarageAge`, `Has{col}`, `"Other"`) are referenced consistently in Task 9's notebook.
- **No placeholders:** every step has real code or a real command; no "add appropriate tests" style steps.
