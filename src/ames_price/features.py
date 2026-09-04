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

    def fit(self, X: pd.DataFrame, y: object = None) -> FeatureEngineer:
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
