# Multi-Store Sales Forecasting & Decision System

An end-to-end retail sales forecasting project using time-series analysis, feature engineering, machine learning, recursive backtesting, and Power BI for business decision support.

## Overview

This project develops a store-level sales forecasting system using historical Walmart retail sales data.

The system analyzes historical demand patterns, engineers time-series features, compares forecasting approaches, evaluates models using rolling recursive backtesting, validates the selected model on an untouched historical holdout, and generates an 8-week sales forecast for 45 stores.

The forecasting results are further analyzed at store and network level and presented through an interactive Power BI dashboard.

## Business Objective

The objective is to support retail planning decisions related to:

- Inventory allocation
- Replenishment planning
- Staffing
- Operational capacity
- Store prioritization
- Holiday demand planning
- Forecast accuracy monitoring

## Dataset

The project uses the Walmart Recruiting Store Sales Forecasting dataset.

### Training Data

| Attribute | Value |
|---|---:|
| Observations | 421,570 |
| Stores | 45 |
| Departments | 81 |
| Target Variable | `Weekly_Sales` |
| Start Date | 2010-02-05 |
| End Date | 2012-10-26 |

### Store Data

| Store Type | Number of Stores |
|---|---:|
| Type A | 22 |
| Type B | 17 |
| Type C | 6 |

### Feature Data

The feature dataset includes:

- Temperature
- Fuel Price
- CPI
- Unemployment
- MarkDown1
- MarkDown2
- MarkDown3
- MarkDown4
- MarkDown5
- Holiday information

## Data Preparation

Department-level sales are combined with store metadata and store-date features.

Sales are aggregated from the department level to the `Store × Week` level.

The resulting prepared dataset contains:

- 6,435 store-week observations
- 45 stores
- 143 weeks

Sales reconciliation is performed after aggregation to ensure that the total historical sales are preserved.

## Time-Series Analysis

The historical store-week data is analyzed for:

- Weekly sales trends
- Monthly seasonality
- Holiday effects
- Store-level sales variation
- Store-type differences
- Autocorrelation

The analysis shows a strong relationship at the 52-week lag, indicating significant year-over-year seasonality.

This supports the use of `lag_52` as an important forecasting feature.

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
- `MarkDown1`
- `MarkDown2`
- `MarkDown3`
- `MarkDown4`
- `MarkDown5`
- `CPI`
- `Unemployment`

Lag features are calculated separately for each store.

Rolling features use shifted historical observations to prevent future information from entering the feature calculations.

## Models

The project compares three forecasting approaches.

### Seasonal Naive

A seasonal benchmark based on historical year-over-year sales.

### Gradient Boosting — Core

A `GradientBoostingRegressor` using historical, calendar, and store-level features.

### Gradient Boosting — Core + External

A Gradient Boosting model using the core features together with external variables.

External variables are evaluated through model comparison rather than being assumed to improve forecasting performance.

## Model Evaluation

### Rolling 8-Week Recursive Backtesting

Four historical rolling windows are used for model comparison.

For each window:

1. The model is trained using observations available before the forecast origin.
2. The next eight weeks are forecast recursively.
3. Each prediction is used when generating the following forecast.
4. Predictions are compared with actual sales.
5. MAE, RMSE, and WAPE are calculated.

This evaluation approach reflects the recursive forecasting strategy used for the final forecast.

## Evaluation Metrics

### MAE

Mean Absolute Error measures the average absolute difference between actual and predicted sales.

```text
MAE = mean(|Actual - Forecast|)
```

### RMSE

Root Mean Squared Error gives greater weight to larger forecasting errors.

```text
RMSE = sqrt(mean((Actual - Forecast)^2))
```

### WAPE

Weighted Absolute Percentage Error is used as the primary business metric.

```text
WAPE = Σ|Actual - Forecast| / Σ|Actual|
```

Lower values indicate better forecasting performance.

## Rolling Backtest Results

Average performance across the four rolling 8-week backtests:

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Gradient Boosting — Core + External | 45,019 | 69,946 | **4.28%** |
| Seasonal Naive | 54,571 | 81,262 | 5.18% |
| Gradient Boosting — Core | 60,982 | 92,711 | 5.79% |

The Gradient Boosting — Core + External model achieved the lowest average WAPE and was selected for the final forecasting stage.

Its WAPE across the four rolling windows ranged from approximately 3.99% to 4.72%.

## Final Holdout Evaluation

After model selection, an untouched historical holdout period was used to evaluate the selected model before final retraining.

### Holdout Period

**2012-09-28 to 2012-10-26**

### Holdout Performance

| Metric | Result |
|---|---:|
| MAE | 47,030 |
| RMSE | 72,007 |
| WAPE | **4.64%** |

The selected model was then retrained using all available historical modeling data.

## Final Forecast

The final model generates an 8-week recursive forecast for all 45 stores.

### Forecast Period

**2012-11-02 to 2012-12-21**

### Forecast Coverage

| Attribute | Value |
|---|---:|
| Stores | 45 |
| Weeks | 8 |
| Store-Week Forecasts | 360 |

### Forecast Results

| Metric | Result |
|---|---:|
| Total Forecasted Sales | 453.85 million |
| Peak Forecast Week | 2012-12-21 |
| Peak Weekly Forecast | 72.86 million |
| Thanksgiving Week Forecast | 65.87 million |
| Highest Forecast Store | Store 4 |
| Store 4 8-Week Forecast | 20.35 million |

## Business Insights

### Demand Concentration

Forecasted sales are concentrated across higher-volume stores.

This supports differentiated planning for inventory, staffing, and operational capacity rather than applying identical policies across all stores.

### Seasonality

The strong 52-week relationship indicates significant year-over-year seasonality.

Historical seasonal demand is therefore an important component of the forecasting system.

### Holiday Demand

Holiday periods influence sales patterns and should be considered during inventory and staffing planning.

The final forecast includes the Thanksgiving period and the year-end demand period.

### External Variables

External variables improved rolling backtest performance when added to the core forecasting features.

They were therefore retained in the selected model.

### Store-Level Accuracy

Store-level WAPE provides a way to identify locations where forecasting performance is relatively weaker.

These stores can receive additional analysis or store-specific forecasting improvements.

## Power BI Dashboard

The project includes an interactive Power BI dashboard for analyzing sales performance, forecast accuracy, store behavior, seasonality, and business planning implications.

### Dashboard Pages

1. **Overview**
   - Total sales
   - Forecast sales
   - Forecast accuracy
   - Store coverage
   - Overall performance

2. **Store Performance & Forecast Accuracy**
   - Store sales
   - Forecast accuracy
   - WAPE
   - Forecast errors
   - Store-level performance

3. **Store Deep-Dive**
   - Detailed individual store analysis
   - Historical performance
   - Forecast performance

4. **Store Ranking & Type Comparison**
   - Store rankings
   - Store types
   - Sales performance
   - Forecast accuracy

5. **Seasonality & Holiday Insights**
   - Monthly patterns
   - Seasonal demand
   - Holiday effects
   - Forecast trends

6. **Business Decision Support**
   - Inventory planning
   - Staffing
   - Capacity planning
   - Store prioritization

## Project Structure

```text
Multi-Store-Sales-Forecasting-Decision-System/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_time_series_analysis.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_baseline_and_forecasting.ipynb
│   ├── 06_backtesting_and_model_comparison.ipynb
│   └── 07_final_forecast_and_business_insights.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py
│   ├── features.py
│   ├── forecasting.py
│   └── evaluation.py
│
├── tests/
│   ├── test_data_validation.py
│   ├── test_features.py
│   ├── test_forecasting.py
│   └── test_evaluation.py
│
├── docs/
│   ├── methodology.md
│   ├── model_evaluation.md
│   ├── metric_definitions.md
│   └── business_insights.md
│
├── reports/
│   ├── model_evaluation_summary.csv
│   ├── store_error_summary.csv
│   └── forecast_summary.csv
│
├── powerbi/
│   └── Multi-Store-Sales-Forecasting-Decision-System.pbix
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Statistical Analysis | SciPy, Statsmodels |
| Machine Learning | Scikit-learn |
| Forecasting | Gradient Boosting Regressor |
| Business Intelligence | Microsoft Power BI |
| Development Environment | Jupyter Lab |

## Key Project Outputs

### Prepared Dataset

`data/processed/store_week_features.csv`

Final store-week dataset containing the engineered forecasting features.

### Final Forecast

`data/processed/future_forecast.csv`

Store-level forecasts for the final 8-week forecasting horizon.

### Power BI Dashboard

`powerbi/Multi-Store-Sales-Forecasting-Decision-System.pbix`

Interactive dashboard for forecasting analysis and business decision support.

## How to Run

### Clone the Repository

```bash
git clone <repository-url>
cd Multi-Store-Sales-Forecasting-Decision-System
```

### Create a Virtual Environment

```bash
python -m venv .venv
```

### Activate the Environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch Jupyter Lab

```bash
jupyter lab
```

### Run the Notebooks

Run the notebooks in the following order:

1. `01_data_understanding.ipynb`
2. `02_data_preparation.ipynb`
3. `03_time_series_analysis.ipynb`
4. `04_feature_engineering.ipynb`
5. `05_baseline_and_forecasting.ipynb`
6. `06_backtesting_and_model_comparison.ipynb`
7. `07_final_forecast_and_business_insights.ipynb`

## Model Limitations

- The final forecast covers an 8-week horizon.
- Recursive forecasting can allow prediction errors to propagate across future weeks.
- Store-level demand varies substantially, so aggregate accuracy does not guarantee equal accuracy across all stores.
- External variables have different levels of availability and predictive value.
- The final forecasting stage operates at the store-week level rather than directly forecasting individual departments.
- The model does not currently provide probabilistic prediction intervals.
- Store-specific models could potentially improve performance for locations with higher forecast error.

## Future Improvements

- Store-specific forecasting models
- Department-level forecasting
- Hyperparameter optimization
- Comparison with LightGBM or XGBoost
- Hierarchical forecasting across store and department levels
- Probabilistic forecasting
- Prediction intervals
- Automated forecast monitoring
- Model drift detection
- Additional promotion and event features
- Automated Power BI refresh pipelines

## Key Results

| Metric | Result |
|---|---:|
| Stores Forecasted | **45** |
| Forecast Horizon | **8 weeks** |
| Store-Week Forecasts | **360** |
| Average Rolling Backtest WAPE | **4.28%** |
| Final Holdout WAPE | **4.64%** |
| Total Forecasted Sales | **453.85 million** |
| Peak Forecast Week | **2012-12-21** |
| Peak Weekly Forecast | **72.86 million** |
| Highest Forecast Store | **Store 4** |
| Store 4 8-Week Forecast | **20.35 million** |

## Conclusion

The project demonstrates a complete retail forecasting workflow from raw data preparation to model evaluation, final forecasting, and business decision support.

The selected Gradient Boosting — Core + External model achieved an average rolling backtest WAPE of **4.28%** and a final historical holdout WAPE of **4.64%**.

The final system produces store-level forecasts across 45 stores and an 8-week horizon, with the results integrated into Power BI for business analysis and operational planning.