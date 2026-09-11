# Methodology

## Objective

Build a multi-store weekly sales forecasting system that supports store-level demand planning and business decision-making.

## Data Preparation

The project combines four source datasets:

- `train.csv` — historical store-department weekly sales
- `features.csv` — calendar, holiday, economic, and promotional variables
- `stores.csv` — store type and size information
- `test.csv` — future Store-Department-Date combinations without sales targets

Historical department-level sales are aggregated to the Store-Week level for network-level forecasting.

The final Store-Week dataset contains:

- Store
- Date
- Weekly_Sales
- IsHoliday
- store attributes
- calendar variables
- economic and promotional variables

## Time-Series Structure

The forecasting dataset is ordered chronologically within each store.

The analysis identified strong weekly dependence and pronounced yearly seasonality. The 52-week autocorrelation was particularly strong, supporting year-over-year lag features.

## Feature Engineering

### Core Features

- `lag_1`
- `lag_2`
- `lag_52`
- `rolling_mean_4`
- `rolling_std_4`
- `WeekOfYear`
- `Month`
- `HolidayName`
- `Type`
- `Size`

### External Features

- `Temperature`
- `Fuel_Price`
- `MarkDown1`–`MarkDown5`
- `CPI`
- `Unemployment`

External variables were evaluated separately and retained because they improved forecasting performance during model comparison.

## Leakage Prevention

All lag features are calculated within each store.

Rolling statistics use a one-period shift before aggregation so that the current week's sales are not included in the feature calculation.

Model development uses chronological splits rather than random sampling.

The final five-week holdout period is kept separate from model development and is evaluated before final retraining.

## Forecasting Approach

Two forecasting approaches are evaluated:

1. Seasonal Naive
2. Gradient Boosting Regressor

The Seasonal Naive model provides a year-over-year benchmark using the corresponding value from 52 weeks earlier.

The Gradient Boosting model uses engineered time-series, calendar, store, and selected external features.

## Recursive Forecasting

The final Gradient Boosting forecast is generated recursively across the eight-week horizon.

After each forecasted week, the prediction becomes available as historical input for subsequent lag and rolling features.

This prevents the future target values from being used during forecasting.

## Model Selection

Model selection is based on rolling eight-week backtesting.

Four chronological backtest windows are used during development.

The model with the strongest average WAPE performance is selected for final forecasting.

## Final Evaluation

A separate five-week holdout from:

`2012-09-28` to `2012-10-26`

is used for final model evaluation.

After holdout evaluation, the selected model is retrained using all available historical modeling data through `2012-10-26`.

The final forecast covers:

`2012-11-02` to `2012-12-21`

for 45 stores.