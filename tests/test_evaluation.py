import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    calculate_mae,
    calculate_rmse,
    calculate_wape,
    calculate_metrics,
    calculate_store_metrics,
    evaluate_forecast,
)


def test_mae():
    actual = [100, 200, 300]
    forecast = [90, 210, 300]

    assert calculate_mae(
        actual,
        forecast,
    ) == pytest.approx(
        20 / 3
    )


def test_rmse():
    actual = [100, 200]
    forecast = [90, 210]

    expected = np.sqrt(
        (100 + 100) / 2
    )

    assert calculate_rmse(
        actual,
        forecast,
    ) == pytest.approx(expected)


def test_wape():
    actual = [100, 200, 300]
    forecast = [90, 210, 300]

    expected = 20 / 600

    assert calculate_wape(
        actual,
        forecast,
    ) == pytest.approx(expected)


def test_wape_zero_denominator():
    with pytest.raises(ValueError):
        calculate_wape(
            [0, 0],
            [0, 0],
        )


def test_calculate_metrics():
    metrics = calculate_metrics(
        [100, 200],
        [90, 210],
    )

    assert set(metrics.keys()) == {
        "MAE",
        "RMSE",
        "WAPE",
    }


def test_forecast_evaluation():
    actual = pd.DataFrame(
        {
            "Store": [1, 1],
            "Date": pd.to_datetime(
                ["2012-01-06", "2012-01-13"]
            ),
            "Weekly_Sales": [100.0, 200.0],
        }
    )

    forecast = pd.DataFrame(
        {
            "Store": [1, 1],
            "Date": pd.to_datetime(
                ["2012-01-06", "2012-01-13"]
            ),
            "Forecast": [90.0, 210.0],
        }
    )

    metrics, merged = evaluate_forecast(
        actual,
        forecast,
    )

    assert len(merged) == 2
    assert metrics["WAPE"] == pytest.approx(
        20 / 300
    )


def test_store_metrics():
    df = pd.DataFrame(
        {
            "Store": [1, 1, 2, 2],
            "Weekly_Sales": [
                100.0,
                200.0,
                300.0,
                400.0,
            ],
            "Forecast": [
                90.0,
                210.0,
                290.0,
                410.0,
            ],
        }
    )

    result = calculate_store_metrics(df)

    assert len(result) == 2
    assert set(result["Store"]) == {1, 2}