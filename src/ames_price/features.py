"""Feature engineering: the judgment calls EDA flagged but left open."""

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
    """Cascading merge to satisfy a minimum per-level count.

    Repeatedly finds the globally thinnest level and merges it into
    whichever adjacent documented-order level is more populous, until
    every remaining group clears min_count (or only one group is left).
    Keeps the survivor's own label, never invents a new one -- Encoder's
    ORDER dict stays valid unchanged.
    """
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
    """The judgment calls EDA flagged but left open.

    Structural-identity resolution, derived features, has-X flags, and
    rare-level bucketing.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> FeatureEngineer:
        """Fit rare-level merge maps and nominal majority labels."""
        self.ordinal_merge_maps_ = {}
        for col in _ORDINAL_BUCKET_COLS:
            if col not in X.columns:
                continue
            order = ORDER[col]
            # "None" means the feature doesn't exist at all -- it must
            # never be merged with a real (if thinly-populated) quality
            # rating, so it's excluded from bucketing and mapped to
            # itself rather than left to the cascading merge below.
            has_none_level = order[0] == "None"
            mergeable_order = order[1:] if has_none_level else order
            merge_map = _build_merge_map(
                X[col], mergeable_order, _MIN_LEVEL_COUNT
            )
            if has_none_level:
                merge_map["None"] = "None"
            self.ordinal_merge_maps_[col] = merge_map
        self.nominal_majority_ = {
            col: X[col].mode().iloc[0]
            for col in _NOMINAL_OTHER_BUCKET_COLS
            if col in X.columns
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply structural drops, derived features, has-X flags, bucketing."""
        X = X.copy()

        X = X.drop(columns=["1stFlrSF", "BsmtUnfSF"], errors="ignore")

        if "GarageType" in X.columns:
            has_garage = X["GarageType"] != "None"
            X["GarageAge"] = np.where(
                has_garage, X["GarageYrBlt"] - X["YearBuilt"], 0
            )
            X["HasGarage"] = has_garage.astype(int)

        for col in _HAS_KEEP_MAGNITUDE_COLS:
            if col in X.columns:
                X[f"Has{col}"] = (X[col] > 0).astype(int)

        for col in _HAS_DROP_MAGNITUDE_COLS:
            if col in X.columns:
                X[f"Has{col}"] = (X[col] > 0).astype(int)
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
