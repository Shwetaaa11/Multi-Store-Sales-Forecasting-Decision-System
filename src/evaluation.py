import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


def calculate_mae(
    actual,
    forecast
):
    """
    Calculate Mean Absolute Error.
    """

    return mean_absolute_error(
        actual,
        forecast
    )


def calculate_rmse(
    actual,
    forecast
):
    """
    Calculate Root Mean Squared Error.
    """

    return np.sqrt(
        mean_squared_error(
            actual,
            forecast
        )
    )


def calculate_wape(
    actual,
    forecast
):
    """
    Calculate Weighted Absolute Percentage Error.

    WAPE = sum(|Actual - Forecast|) / sum(|Actual|)
    """

    actual = np.asarray(
        actual,
        dtype=float
    )

    forecast = np.asarray(
        forecast,
        dtype=float
    )

    denominator = np.sum(
        np.abs(actual)
    )

    if denominator == 0:
        raise ValueError(
            "WAPE cannot be calculated when "
            "the sum of absolute actual values is zero."
        )

    numerator = np.sum(
        np.abs(
            actual - forecast
        )
    )

    return numerator / denominator


def calculate_metrics(
    actual,
    forecast
):
    """
    Calculate MAE, RMSE, and WAPE.
    """

    return {
        "MAE": calculate_mae(
            actual,
            forecast
        ),
        "RMSE": calculate_rmse(
            actual,
            forecast
        ),
        "WAPE": calculate_wape(
            actual,
            forecast
        ),
    }


def evaluate_forecast(
    actual_dataframe,
    forecast_dataframe,
    actual_column="Weekly_Sales",
    forecast_column="Forecast",
):
    """
    Evaluate forecast predictions against actual values.

    The two datasets must contain matching Store-Date keys.
    """

    actual = actual_dataframe.copy()
    forecast = forecast_dataframe.copy()

    actual["Date"] = pd.to_datetime(
        actual["Date"]
    )

    forecast["Date"] = pd.to_datetime(
        forecast["Date"]
    )

    required_actual = [
        "Store",
        "Date",
        actual_column,
    ]

    required_forecast = [
        "Store",
        "Date",
        forecast_column,
    ]

    for column in required_actual:
        if column not in actual.columns:
            raise ValueError(
                f"Missing actual column: {column}"
            )

    for column in required_forecast:
        if column not in forecast.columns:
            raise ValueError(
                f"Missing forecast column: {column}"
            )

    merged = actual[
        required_actual
    ].merge(
        forecast[
            required_forecast
        ],
        on=["Store", "Date"],
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(actual):
        raise ValueError(
            "Forecast does not contain predictions "
            "for every actual Store-Date record."
        )

    metrics = calculate_metrics(
        merged[actual_column],
        merged[forecast_column],
    )

    return metrics, merged


def calculate_store_metrics(
    dataframe,
    actual_column="Weekly_Sales",
    forecast_column="Forecast",
):
    """
    Calculate forecast metrics separately for each store.
    """

    results = []

    for store, group in dataframe.groupby(
        "Store"
    ):

        metrics = calculate_metrics(
            group[actual_column],
            group[forecast_column],
        )

        results.append(
            {
                "Store": store,
                **metrics,
            }
        )

    return pd.DataFrame(
        results
    ).sort_values(
        "Store"
    ).reset_index(drop=True)


def compare_models(
    results_dataframe,
    model_column="Model",
    metric_column="WAPE",
):
    """
    Compare models using the mean evaluation metric.
    """

    comparison = (
        results_dataframe
        .groupby(model_column)[metric_column]
        .mean()
        .sort_values()
        .reset_index()
    )

    return comparison


def select_best_model(
    results_dataframe,
    model_column="Model",
    metric_column="WAPE",
):
    """
    Select the model with the lowest average evaluation metric.
    """

    comparison = compare_models(
        results_dataframe,
        model_column=model_column,
        metric_column=metric_column,
    )

    return comparison.iloc[0][
        model_column
    ]