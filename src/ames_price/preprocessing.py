from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ames_price.constants import ORDER, ORDINAL_COLS

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


class Imputer(BaseEstimator, TransformerMixin):
    """Fills every NaN using statistics fit on training data only.

    Structural-absence NaNs (no pool/garage/basement/veneer) get a fixed
    "None"/0 fill. A handful of rows have the feature present but one
    sub-field still missing (e.g. a real garage with GarageQual
    unrecorded) -- those get a statistic fit from other rows that do
    have the feature, not the structural default.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> Imputer:
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

        self.lotfrontage_by_neighborhood_ = X.groupby("Neighborhood")[
            "LotFrontage"
        ].median()
        self.lotfrontage_global_median_ = X["LotFrontage"].median()
        self.mode_fills_ = {
            col: X[col].mode().iloc[0] for col in _REAL_GAP_MODE_COLS
        }

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

        neighborhood_fill = X["Neighborhood"].map(
            self.lotfrontage_by_neighborhood_
        )
        X["LotFrontage"] = (
            X["LotFrontage"]
            .fillna(neighborhood_fill)
            .fillna(self.lotfrontage_global_median_)
        )

        for col, fill in self.mode_fills_.items():
            X[col] = X[col].fillna(fill)

        assert not X.isna().any().any(), (  # noqa: S101
            f"Imputer left NaNs in: {X.columns[X.isna().any()].tolist()}"
        )
        return X


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

    def fit(self, X: pd.DataFrame, y: object = None) -> Encoder:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for col in ORDINAL_COLS:
            if col in X.columns:
                X[col] = X[col].map(
                    {level: i for i, level in enumerate(ORDER[col])}  # type: ignore[var-annotated,arg-type]
                )

        for col in _LOG1P_COLS:
            if col in X.columns:
                X[col] = np.log1p(X[col])

        return X
