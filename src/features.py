import numpy as np
import pandas as pd


CORE_FEATURES = [
    "lag_1",
    "lag_2",
    "lag_52",
    "rolling_mean_4",
    "rolling_std_4",
    "WeekOfYear",
    "Month",
    "HolidayName",
    "Type",
    "Size",
]


EXTERNAL_FEATURES = [
    "Temperature",
    "Fuel_Price",
    "MarkDown1",
    "MarkDown2",
    "MarkDown3",
    "MarkDown4",
    "MarkDown5",
    "CPI",
    "Unemployment",
]


def create_lag_features(
    dataframe,
    target_column="Weekly_Sales",
    group_column="Store",
    lags=(1, 2, 52)
):
    """
    Create store-level lag features.
    """

    df = dataframe.copy()

    df = df.sort_values(
        [group_column, "Date"]
    ).reset_index(drop=True)

    for lag in lags:
        df[f"lag_{lag}"] = (
            df.groupby(group_column)[target_column]
            .shift(lag)
        )

    return df


def create_rolling_features(
    dataframe,
    target_column="Weekly_Sales",
    group_column="Store",
    window=4
):
    """
    Create leakage-safe rolling mean and standard deviation.

    The target is shifted by one period before rolling so that
    the current week's target is not included.
    """

    df = dataframe.copy()

    df = df.sort_values(
        [group_column, "Date"]
    ).reset_index(drop=True)

    shifted = (
        df.groupby(group_column)[target_column]
        .shift(1)
    )

    df[f"rolling_mean_{window}"] = (
        shifted
        .groupby(df[group_column])
        .rolling(window=window, min_periods=window)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df[f"rolling_std_{window}"] = (
        shifted
        .groupby(df[group_column])
        .rolling(window=window, min_periods=window)
        .std()
        .reset_index(level=0, drop=True)
    )

    return df


def create_calendar_features(
    dataframe,
    date_column="Date"
):
    """
    Create calendar-based forecasting features.
    """

    df = dataframe.copy()

    df[date_column] = pd.to_datetime(
        df[date_column]
    )

    df["WeekOfYear"] = (
        df[date_column]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    df["Month"] = (
        df[date_column]
        .dt.month
    )

    return df


def add_holiday_names(
    dataframe,
    date_column="Date"
):
    """
    Map known Walmart holiday dates to holiday names.
    """

    df = dataframe.copy()

    df[date_column] = pd.to_datetime(
        df[date_column]
    )

    holiday_map = {
        "2010-02-12": "Super Bowl",
        "2010-09-10": "Labor Day",
        "2010-11-26": "Thanksgiving",
        "2010-12-31": "Christmas",
        "2011-02-11": "Super Bowl",
        "2011-09-09": "Labor Day",
        "2011-11-25": "Thanksgiving",
        "2011-12-30": "Christmas",
        "2012-02-10": "Super Bowl",
        "2012-09-07": "Labor Day",
        "2012-11-23": "Thanksgiving",
        "2012-12-28": "Christmas",
    }

    holiday_map = {
        pd.Timestamp(date): name
        for date, name in holiday_map.items()
    }

    df["HolidayName"] = (
        df[date_column]
        .map(holiday_map)
        .fillna("None")
    )

    return df


def create_store_week_features(
    dataframe
):
    """
    Create the complete store-week forecasting feature set.
    """

    df = dataframe.copy()

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = df.sort_values(
        ["Store", "Date"]
    ).reset_index(drop=True)

    df = create_lag_features(df)

    df = create_rolling_features(df)

    df = create_calendar_features(df)

    df = add_holiday_names(df)

    return df


def get_modeling_dataset(
    dataframe,
    feature_columns=None
):
    """
    Return rows containing all required modeling features.
    """

    if feature_columns is None:
        feature_columns = CORE_FEATURES

    required_columns = [
        "Store",
        "Date",
        "Weekly_Sales",
        *feature_columns,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    df = dataframe.dropna(
        subset=feature_columns
    ).copy()

    return df.reset_index(drop=True)


def validate_lag_features(
    dataframe
):
    """
    Validate that lag features do not contain
    values from the current target period.
    """

    df = dataframe.sort_values(
        ["Store", "Date"]
    ).copy()

    for lag in [1, 2, 52]:

        expected = (
            df.groupby("Store")["Weekly_Sales"]
            .shift(lag)
        )

        mask = df[f"lag_{lag}"].notna()

        if not np.allclose(
            df.loc[mask, f"lag_{lag}"],
            expected.loc[mask],
            equal_nan=True
        ):
            raise AssertionError(
                f"lag_{lag} validation failed."
            )

    return True