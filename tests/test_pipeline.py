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
    assert train_out["GarageAge"].between(-30, 150).all()
    assert test_out["GarageAge"].between(-30, 150).all()
