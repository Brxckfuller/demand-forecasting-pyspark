# Demand Forecasting with PySpark

## Project Summary

This project explores demand forecasting using transaction data from an online retail store. The goal was to investigate whether machine learning models could improve on a simple forecasting baseline when predicting future sales.

The project was built using PySpark and follows a typical machine learning workflow: data cleaning, feature engineering, model training, evaluation, and result comparison.

## Dataset

The project uses the Online Retail dataset from the UCI Machine Learning Repository.

The dataset contains over 500,000 retail transactions, including invoice information, product descriptions, quantities purchased, unit prices, customer IDs, and transaction dates.

After cleaning the data, sales were aggregated into weekly totals to create a forecasting problem.

## Data Preparation

The cleaning process removed missing product codes, missing transaction dates, missing quantities, returns, cancelled orders, and transactions with non-positive prices.

## Feature Engineering

The forecasting dataset used lag-based features:

- Previous week's sales
- Sales from two weeks ago
- Sales from three weeks ago
- Sales from four weeks ago
- Four-week rolling average

## Models Evaluated

The following models were compared:

- Baseline forecast using previous week's sales
- Linear Regression
- Random Forest Regressor
- Gradient Boosted Trees Regressor

## Results

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Baseline | 22,811.57 | 30,533.65 | 23.19% |
| Linear Regression | 34,041.60 | 44,839.67 | 18.37% |
| Random Forest | 53,737.69 | 62,447.57 | 29.70% |
| Gradient Boosted Trees | 53,813.57 | 62,507.84 | 29.86% |

## Discussion

The simple baseline model achieved the lowest MAE and RMSE. This suggests that weekly demand in this dataset has strong short-term autocorrelation, meaning the previous week's sales are a strong signal for the following week.

The machine learning models did not outperform the baseline. This is likely because the final weekly forecasting dataset contained only 53 observations after aggregation. With such a small time series, more complex models such as Random Forests and Gradient Boosted Trees had limited data to learn from.

This was a useful result because it shows that model complexity does not automatically lead to better forecasting performance.

## Technologies Used

- Python
- PySpark
- Spark MLlib
- Time series feature engineering
- Machine learning model evaluation

## Future Improvements

Possible extensions include:

- Forecasting demand at the product/SKU level
- Adding holiday and seasonal features
- Using a longer historical dataset
- Comparing against ARIMA or Prophet
- Building a dashboard for forecast results

## Author

Brock Fuller  
Master of Artificial Intelligence, RMIT University