# Metric Definitions

## WAPE

Weighted Absolute Percentage Error measures total absolute forecast error relative to total absolute actual sales.

$$
WAPE =
\frac{\sum |Actual - Forecast|}
{\sum |Actual|}
$$

Lower WAPE indicates better forecasting performance.

WAPE is the primary model-selection metric because it evaluates aggregate forecast error relative to the scale of actual sales.

## MAE

Mean Absolute Error measures the average absolute difference between actual and forecast sales.

$$
MAE =
\frac{1}{n}
\sum |Actual - Forecast|
$$

Lower MAE indicates better performance.

MAE is expressed in the same units as weekly sales.

## RMSE

Root Mean Squared Error gives greater weight to larger forecast errors.

$$
RMSE =
\sqrt{
\frac{1}{n}
\sum (Actual - Forecast)^2
}
$$

Lower RMSE indicates better performance.

RMSE is useful for identifying models that produce large individual forecast errors.

## Metric Usage

| Metric | Primary Use |
|---|---|
| WAPE | Model comparison and selection |
| MAE | Average forecast error |
| RMSE | Sensitivity to large errors |

All three metrics are reported for the final evaluation.