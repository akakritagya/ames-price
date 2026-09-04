from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ames_price.constants import ORDER

_HAS_KEEP_MAGNITUDE_COLS = [
    "OpenPorchSF",
    "MasVnrArea",
    "WoodDeckSF",
    "2ndFlrSF",
    "EnclosedPorch",
]
_HAS_DROP_MAGNITUDE_COLS = ["PoolArea", "LowQualFinSF", "3SsnPorch"]

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

    return {
        level: rep
        for group, rep in zip(members, reps, strict=True)
        for level in group
    }


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Structural-identity resolution, derived features, has-X flags, and
    rare-level bucketing -- the judgment calls EDA flagged but left open.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> FeatureEngineer:
        self.ordinal_merge_maps_ = {
            col: _build_merge_map(X[col], ORDER[col], _MIN_LEVEL_COUNT)  # type: ignore[arg-type]
            for col in _ORDINAL_BUCKET_COLS
            if col in X.columns
        }
        self.nominal_majority_ = {
            col: X[col].mode().iloc[0]
            for col in _NOMINAL_OTHER_BUCKET_COLS
            if col in X.columns
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        X = X.drop(columns=["1stFlrSF", "BsmtUnfSF"], errors="ignore")

        if "GarageType" in X.columns:
            X["HasGarage"] = X["GarageType"] != "None"
            X["GarageAge"] = np.where(
                X["HasGarage"], X["GarageYrBlt"] - X["YearBuilt"], 0
            )

        for col in _HAS_KEEP_MAGNITUDE_COLS:
            if col in X.columns:
                X[f"Has{col}"] = X[col] > 0

        for col in _HAS_DROP_MAGNITUDE_COLS:
            if col in X.columns:
                X[f"Has{col}"] = X[col] > 0
                X = X.drop(columns=[col])

        for col, merge_map in self.ordinal_merge_maps_.items():
            if col in X.columns:
                X[col] = X[col].map(merge_map)

        for col in _NOMINAL_OTHER_BUCKET_COLS:
            if col in X.columns:
                majority = self.nominal_majority_[col]
                X[col] = np.where(X[col] == majority, majority, "Other")

        X = X.drop(columns=["Utilities"], errors="ignore")

        return X
