import pandas as pd
import pytest

from src.data_utils import (
    validate_required_columns,
    validate_unique_keys,
)


def test_required_columns_pass():
    df = pd.DataFrame(
        {
            "Store": [1, 2],
            "Date": pd.to_datetime(
                ["2012-01-06", "2012-01-13"]
            ),
            "Weekly_Sales": [100.0, 200.0],
        }
    )

    assert validate_required_columns(
        df,
        ["Store", "Date", "Weekly_Sales"],
    )


def test_required_columns_fail():
    df = pd.DataFrame(
        {
            "Store": [1, 2],
            "Date": pd.to_datetime(
                ["2012-01-06", "2012-01-13"]
            ),
        }
    )

    with pytest.raises(ValueError):
        validate_required_columns(
            df,
            ["Store", "Date", "Weekly_Sales"],
        )


def test_unique_key_pass():
    df = pd.DataFrame(
        {
            "Store": [1, 1],
            "Date": pd.to_datetime(
                ["2012-01-06", "2012-01-13"]
            ),
        }
    )

    assert validate_unique_keys(
        df,
        ["Store", "Date"],
    )


def test_unique_key_fail():
    df = pd.DataFrame(
        {
            "Store": [1, 1],
            "Date": pd.to_datetime(
                ["2012-01-06", "2012-01-06"]
            ),
        }
    )

    with pytest.raises(ValueError):
        validate_unique_keys(
            df,
            ["Store", "Date"],
        )