from ames_price.constants import (
    IDENTIFIER,
    NOMINAL_COLS,
    NUMERIC_COLS,
    ORDER,
    ORDINAL_COLS,
    TARGET,
)


def test_column_groups_are_disjoint_and_cover_expected_counts():
    assert len(NUMERIC_COLS) == 33
    assert len(ORDINAL_COLS) == 21
    assert len(NOMINAL_COLS) == 25
    all_cols = NUMERIC_COLS + ORDINAL_COLS + NOMINAL_COLS
    assert len(all_cols) == len(set(all_cols))


def test_order_covers_exactly_the_ordinal_columns():
    assert set(ORDER.keys()) == set(ORDINAL_COLS)


def test_identifier_and_target_are_singletons():
    assert IDENTIFIER == ["Id"]
    assert TARGET == ["SalePrice"]
