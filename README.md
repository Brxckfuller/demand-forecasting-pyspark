# Demand Forecasting with PySpark

This project forecasts weekly product demand using PySpark and machine learning.

The goal was to compare several forecasting models against a simple baseline and evaluate whether additional feature engineering improved prediction accuracy on historical sales data.

---

## Features

- Data cleaning and preprocessing with PySpark
- Weekly demand aggregation
- Lag and rolling-average feature engineering
- Baseline forecasting model
- Linear Regression, Random Forest and Gradient Boosted Trees
- Model comparison using MAE, RMSE and MAPE
- Automated saving of evaluation results

---

## Dataset

The project uses the Online Retail dataset from the UCI Machine Learning Repository.

The raw transaction data includes:

- Invoice number
- Product code
- Quantity sold
- Invoice date
- Unit price
- Customer ID
- Country

After preprocessing and weekly aggregation, the final modelling dataset contained **53 weekly observations**.

---

## Technologies

- Python
- PySpark
- Pandas
- Spark MLlib
- Matplotlib

---

## Project Structure

```text
src/
│
├── main.py
├── spark_session.py
├── data_cleaning.py
├── feature_engineering.py
├── forecasting.py
├── model_training.py
├── weekly_model_training.py
├── weekly_features.py
├── save_results.py
└── eda.py
```

---

## Running the Project

Clone the repository

```bash
git clone https://github.com/Brxckfuller/demand-forecasting-pyspark.git

cd demand-forecasting-pyspark
```

Install dependencies

```bash
pip install -r requirements.txt
```

Place the Online Retail dataset in

```text
data/raw/online_retail.csv
```

Run the forecasting pipeline

```bash
python src/main.py
```

---

## Models Compared

- Previous-week baseline
- Linear Regression
- Random Forest Regressor
- Gradient Boosted Trees Regressor

---

## Results

| Model | MAE | RMSE | MAPE |
|------|------:|------:|------:|
| Baseline | 22,811.57 | 30,533.65 | 23.19% |
| Linear Regression | 34,041.60 | 44,839.67 | 18.37% |
| Random Forest | 53,737.69 | 62,447.57 | 29.70% |
| Gradient Boosted Trees | 53,813.57 | 62,507.84 | 29.86% |

---

## Key Findings

The simple baseline forecast produced the lowest MAE and RMSE on this dataset.

With only 53 weekly observations available after aggregation, the machine learning models did not outperform the baseline. This highlighted the importance of comparing against simple forecasting methods before adopting more complex models.

---

## Author

Brock Fuller

Master of Artificial Intelligence — RMIT University
