import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DEFAULT_MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "random_state": 42,
}


def build_gradient_boosting_model(
    numeric_features,
    categorical_features,
    model_params=None
):
    """
    Build the Gradient Boosting forecasting pipeline.
    """

    if model_params is None:
        model_params = DEFAULT_MODEL_PARAMS.copy()

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    model = GradientBoostingRegressor(
        **model_params
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


def prepare_model_data(
    dataframe,
    feature_columns,
    target_column="Weekly_Sales"
):
    """
    Prepare X and y for model training.
    """

    df = dataframe.copy()

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in categorical_columns:
        X[column] = X[column].fillna(
            "None"
        ).astype(str)

    return X, y


def train_model(
    dataframe,
    feature_columns,
    numeric_features,
    categorical_features,
    target_column="Weekly_Sales",
    model_params=None
):
    """
    Train a Gradient Boosting forecasting model.
    """

    X, y = prepare_model_data(
        dataframe,
        feature_columns,
        target_column,
    )

    model = build_gradient_boosting_model(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        model_params=model_params,
    )

    model.fit(X, y)

    return model


def recursive_forecast(
    model,
    history,
    future_calendar,
    feature_columns,
    target_column="Weekly_Sales",
):
    """
    Generate recursive multi-step forecasts.

    Each predicted value is appended to the history and is
    therefore available to subsequent lag calculations.
    """

    history = history.copy()
    future_calendar = future_calendar.copy()

    history["Date"] = pd.to_datetime(
        history["Date"]
    )

    future_calendar["Date"] = pd.to_datetime(
        future_calendar["Date"]
    )

    history = history.sort_values(
        ["Store", "Date"]
    ).reset_index(drop=True)

    future_calendar = future_calendar.sort_values(
        ["Store", "Date"]
    ).reset_index(drop=True)

    stores = sorted(
        future_calendar["Store"].unique()
    )

    if len(stores) == 0:
        raise ValueError(
            "Future calendar contains no stores."
        )

    predictions = []

    for date in sorted(
        future_calendar["Date"].unique()
    ):

        current_calendar = future_calendar[
            future_calendar["Date"] == date
        ].copy()

        for _, row in current_calendar.iterrows():

            store = row["Store"]

            store_history = history[
                history["Store"] == store
            ].sort_values("Date")
            
            if store_history.empty:
                raise ValueError(
                    f"No historical data available for Store {store}."
                )

            previous_sales = store_history[
                target_column
            ]

            if len(previous_sales) < 52:
                raise ValueError(
                    f"Insufficient history for lag_52 "
                    f"for Store {store}."
                )

            lag_1 = previous_sales.iloc[-1]
            lag_2 = previous_sales.iloc[-2]
            lag_52 = previous_sales.iloc[-52:]

            recent_values = previous_sales.iloc[-4:]

            rolling_mean_4 = (
                recent_values.mean()
            )

            rolling_std_4 = (
                recent_values.std()
            )

            feature_row = row.to_dict()

            feature_row["lag_1"] = lag_1
            feature_row["lag_2"] = lag_2
            feature_row["lag_52"] = lag_52.iloc[-1]
            feature_row["rolling_mean_4"] = rolling_mean_4
            feature_row["rolling_std_4"] = rolling_std_4

            prediction_data = pd.DataFrame(
                [feature_row]
            )

            prediction_data = prediction_data[
                feature_columns
            ].copy()

            categorical_columns = (
                prediction_data
                .select_dtypes(
                    include=["object", "category"]
                )
                .columns
            )

            for column in categorical_columns:
                prediction_data[column] = (
                    prediction_data[column]
                    .fillna("None")
                    .astype(str)
                )

            prediction = float(
                model.predict(
                    prediction_data
                )[0]
            )

            predictions.append(
                {
                    "Store": store,
                    "Date": date,
                    "Forecast": prediction,
                }
            )

            history = pd.concat(
                [
                    history,
                    pd.DataFrame(
                        [
                            {
                                "Store": store,
                                "Date": date,
                                target_column: prediction,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    return pd.DataFrame(
        predictions
    )


def seasonal_naive_forecast(
    history,
    future_dates,
    store_column="Store",
    date_column="Date",
    target_column="Weekly_Sales",
    seasonal_lag=52,
):
    """
    Generate seasonal naive forecasts using the value from
    the corresponding seasonal lag.
    """

    history = history.copy()
    history[date_column] = pd.to_datetime(
        history[date_column]
    )

    future_dates = pd.to_datetime(
        future_dates
    )

    history = history.sort_values(
        [store_column, date_column]
    )

    results = []

    for store in sorted(
        history[store_column].unique()
    ):

        store_history = history[
            history[store_column] == store
        ].set_index(date_column)

        for date in future_dates:

            seasonal_date = (
                date
                - pd.Timedelta(
                    weeks=seasonal_lag
                )
            )

            if seasonal_date not in store_history.index:
                raise ValueError(
                    f"Seasonal lag unavailable for "
                    f"Store {store}, Date {date}."
                )

            forecast = store_history.loc[
                seasonal_date,
                target_column
            ]

            results.append(
                {
                    "Store": store,
                    "Date": date,
                    "Forecast": forecast,
                }
            )

    return pd.DataFrame(
        results
    )