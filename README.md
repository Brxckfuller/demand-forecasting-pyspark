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

## How to Run

### Step 1: Clone the repository

```bash
git clone https://github.com/Brxckfuller/demand-forecasting-pyspark.git
cd demand-forecasting-pyspark
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Add the dataset

Place the Online Retail dataset in:

```text
data/raw/online_retail.csv
```

### Step 4: Run the forecasting pipeline

```bash
python src/main.py
```

### Step 5: View results

The pipeline will:

- Clean and validate transaction data
- Aggregate sales into weekly demand
- Create lag and rolling-average features
- Generate a baseline forecast
- Train Linear Regression, Random Forest and Gradient Boosted Tree models
- Compare MAE, RMSE and MAPE across models

Model comparison results are saved automatically to:

```text
data/outputs/model_comparison/
```

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
``` 

## Author

Brock Fuller

Master of Artificial Intelligence – RMIT University
