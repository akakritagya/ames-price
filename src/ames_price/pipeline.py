"""Composes Imputer, FeatureEngineer, and Encoder into one sklearn Pipeline."""

from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ames_price.features import FeatureEngineer
from ames_price.preprocessing import Encoder, Imputer


def build_pipeline() -> Pipeline:
    """Build the impute -> engineer -> encode -> scale pipeline."""
    return Pipeline(
        [
            ("impute", Imputer()),
            ("engineer", FeatureEngineer()),
            ("encode", Encoder()),
            ("scale", StandardScaler().set_output(transform="pandas")),
        ]
    )
