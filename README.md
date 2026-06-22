# Demand Forecasting with PySpark

## Overview

This project uses PySpark to forecast weekly demand from e-commerce transaction data.

The aim was to test whether machine learning models could beat a simple forecasting baseline. The pipeline cleans raw transaction data, aggregates sales by week, creates lag features, trains several models, and compares forecast errors.

## Dataset

The project uses the Online Retail dataset from the UCI Machine Learning Repository.

The raw dataset includes:

- invoice numbers
- product codes
- product descriptions
- quantities sold
- invoice dates
- unit prices
- customer IDs
- countries

After cleaning and aggregation, the final forecasting dataset contained 53 weekly observations.

## Approach

The workflow was:

1. Load transaction data with Spark
2. Remove invalid transactions
3. Convert invoice dates to timestamps
4. Aggregate total sales by year and week
5. Create lag and rolling-average features
6. Split data by time
7. Train and evaluate forecasting models

## Features

The model inputs were:

- `Lag_1_Week`
- `Lag_2_Week`
- `Lag_3_Week`
- `Lag_4_Week`
- `Rolling_4_Week_Avg`

The target variable was:

- `TotalSales`

## Models

The following models were compared:

- Baseline forecast: previous week's sales
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

## Takeaway

The baseline model had the lowest MAE and RMSE.

The machine learning models did not improve on the simple previous-week forecast. The most likely reason is the small number of training examples after weekly aggregation. With only 53 weekly observations, the tree-based models did not have enough history to learn stable patterns.

This project was useful because it showed that a simple baseline is not just a formality in forecasting. It can be difficult to beat when recent demand is already a strong predictor.

## Project Structure

```text
src/
├── main.py
├── spark_session.py
├── data_cleaning.py
├── weekly_features.py
├── weekly_forecasting.py
├── weekly_model_training.py
├── save_results.py
├── save_data.py
├── eda.py
├── feature_engineering.py
├── forecasting.py
├── model_training.py
├── model_training_ml.py
└── model_training_rf.py
