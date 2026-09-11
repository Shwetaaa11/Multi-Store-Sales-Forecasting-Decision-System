import pandas as pd
import pytest

from src.forecasting import (
    build_gradient_boosting_model,
    seasonal_naive_forecast,
)


def test_gradient_boosting_model_creation():
    model = build_gradient_boosting_model(
        numeric_features=[
            "lag_1",
            "lag_2",
            "lag_52",
        ],
        categorical_features=[
            "HolidayName",
            "Type",
        ],
    )

    assert model is not None


def test_seasonal_naive_missing_seasonal_date():
    history = pd.DataFrame(
        {
            "Store": [1, 1],
            "Date": pd.to_datetime(
                ["2012-01-06", "2012-01-13"]
            ),
            "Weekly_Sales": [100.0, 200.0],
        }
    )

    future_dates = pd.to_datetime(
        ["2013-02-01"]
    )

    with pytest.raises(ValueError):
        seasonal_naive_forecast(
            history,
            future_dates,
        )