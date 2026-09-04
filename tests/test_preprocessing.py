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
