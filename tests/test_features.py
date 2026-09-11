import numpy as np
import pandas as pd

from src.features import (
    add_holiday_names,
    create_calendar_features,
    create_lag_features,
    create_rolling_features,
    validate_lag_features,
)


def create_sample_data():
    dates = pd.date_range(
        "2010-01-01",
        periods=60,
        freq="7D",
    )

    return pd.DataFrame(
        {
            "Store": [1] * 60,
            "Date": dates,
            "Weekly_Sales": np.arange(
                100,
                160,
                dtype=float,
            ),
        }
    )


def test_lag_features():
    df = create_sample_data()

    result = create_lag_features(df)

    assert result.loc[1, "lag_1"] == 100
    assert result.loc[2, "lag_2"] == 100
    assert result.loc[52, "lag_52"] == 100


def test_lag_features_validation():
    df = create_sample_data()

    result = create_lag_features(df)

    assert validate_lag_features(result)


def test_rolling_features_are_lagged():
    df = create_sample_data()

    result = create_rolling_features(df)

    assert pd.isna(
        result.loc[0, "rolling_mean_4"]
    )

    assert result.loc[4, "rolling_mean_4"] == 101.5


def test_calendar_features():
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2012-01-06"]
            )
        }
    )

    result = create_calendar_features(df)

    assert result.loc[0, "Month"] == 1
    assert result.loc[0, "WeekOfYear"] == 1


def test_holiday_mapping():
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2012-11-23", "2012-11-30"]
            )
        }
    )

    result = add_holiday_names(df)

    assert result.loc[0, "HolidayName"] == "Thanksgiving"
    assert result.loc[1, "HolidayName"] == "None"