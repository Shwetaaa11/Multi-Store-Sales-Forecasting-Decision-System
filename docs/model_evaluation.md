# Model Evaluation

## Evaluation Strategy

Model performance is evaluated using chronological rolling backtests rather than random train-test splits.

Each backtest forecasts eight weeks recursively.

This evaluation structure reflects the intended production forecasting process more closely than one-step prediction.

## Backtest Windows

| Window | Forecast Start | Forecast End |
|---|---|---|
| 1 | 2012-05-04 | 2012-06-22 |
| 2 | 2012-06-01 | 2012-07-20 |
| 3 | 2012-06-29 | 2012-08-17 |
| 4 | 2012-08-03 | 2012-09-21 |

## Model Comparison

| Model | Mean WAPE |
|---|---:|
| Seasonal Naive | 5.18% |
| Gradient Boosting - Core | 5.79% |
| Gradient Boosting - Core + External | 4.28% |

The Gradient Boosting model using the core and external feature set achieved the strongest average backtest performance and was selected for final forecasting.

## Selected Model

`GradientBoostingRegressor`

Parameters:

- `n_estimators = 200`
- `max_depth = 6`
- `learning_rate = 0.05`
- `random_state = 42`

## Feature Importance

The most influential model features were:

| Feature | Importance |
|---|---:|
| `lag_52` | 79.70% |
| `rolling_mean_4` | 17.44% |
| `lag_1` | 1.76% |
| `WeekOfYear` | 0.29% |
| `lag_2` | 0.27% |

The dominance of `lag_52` indicates that year-over-year seasonal behavior is the primary predictive signal in the selected model.

## Final Holdout

The final untouched holdout covers:

`2012-09-28` to `2012-10-26`

Results:

| Metric | Result |
|---|---:|
| WAPE | 4.64% |
| MAE | 47,030 |
| RMSE | 72,007 |

The holdout is evaluated before the selected model is retrained on the full historical modeling dataset.

## Interpretation

The final evaluation indicates that the selected forecasting approach provides useful store-level demand estimates while preserving a realistic chronological evaluation design.