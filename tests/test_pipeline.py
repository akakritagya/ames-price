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
    assert train_out.shape[1] == 237
    assert list(train_out.columns) == list(test_out.columns)
    assert all(np.issubdtype(dtype, np.number) for dtype in train_out.dtypes)
    assert np.isfinite(train_out["GarageAge"]).all()
    assert np.isfinite(test_out["GarageAge"]).all()


def test_pipeline_output_is_standardized():
    df_train = pd.read_csv("data/train.csv")
    df_train = df_train[~df_train["Id"].isin([524, 1299])]

    pipeline = build_pipeline()
    train_out = pipeline.fit_transform(df_train[FEATURE_COLS])

    means = train_out.mean(axis=0)
    stds = train_out.std(axis=0, ddof=0)
    assert np.allclose(means, 0, atol=1e-6)
    # a zero-variance column (none expected, but not guaranteed by this
    # test) would legitimately stay at std 0 rather than 1 -- StandardScaler
    # skips dividing by a zero variance instead of raising or emitting NaN.
    assert np.all(
        np.isclose(stds, 1, atol=1e-6) | np.isclose(stds, 0, atol=1e-6)
    )
