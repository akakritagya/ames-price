"""Column groupings and documented category orders, from data_description.txt.

Single source of truth for the feature-type split used by both the EDA
notebooks and the preprocessing/feature-engineering pipeline.
"""

from typing import Any

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

ORDER: dict[str, list[Any]] = {
    "OverallQual": list(range(1, 11)),
    "OverallCond": list(range(1, 11)),
    "LotShape": ["IR3", "IR2", "IR1", "Reg"],
    "LandSlope": ["Sev", "Mod", "Gtl"],
    "ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
    "KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    "FireplaceQu": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageFinish": ["None", "Unf", "RFn", "Fin"],
    "GarageQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "PavedDrive": ["N", "P", "Y"],
    "PoolQC": ["None", "Fa", "TA", "Gd", "Ex"],
    "Utilities": ["ELO", "NoSeWa", "NoSewr", "AllPub"],
}
