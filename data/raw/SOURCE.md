# Raw Data Source

Dataset: Walmart Recruiting - Store Sales Forecasting
Source: Kaggle (https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)
Downloaded: via Kaggle web UI (late submission acceptance) on the date these files were pulled
License/Terms: Kaggle competition rules (accepted on download)

## Files
- train.csv       — historical weekly sales by Store, Dept, Date
- test.csv        — same structure as train.csv but Weekly_Sales withheld (not used for this project; forecasting will be done on a held-out slice of train.csv instead, since test.csv has no ground truth available)
- features.csv    — Temperature, Fuel_Price, CPI, Unemployment, MarkDown1-5, IsHoliday by Store and Date
- stores.csv      — Store, Type, Size (45 stores)

## Time Range
train.csv: 2010-02-05 to 2012-10-26, weekly granularity (143 weeks, confirmed complete — no missing weeks)
features.csv: 2010-02-05 to 2013-07-26 (extends beyond train.csv to cover the test/forecast period)

## Known Limitations (to verify during validation)
- MarkDown1-5 columns are reported to have significant missing values in the early part of the date range
- test.csv has no target values, so it will not be used for model evaluation in this project