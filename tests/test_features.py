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
