from spark_session import create_spark_session
from data_cleaning import clean_data
from weekly_features import create_weekly_sales
from weekly_forecasting import create_weekly_forecasting_dataset

from weekly_model_training import (
    evaluate_baseline,
    train_linear_regression_weekly,
    train_random_forest_weekly,
    train_gbt_weekly
)

from save_results import save_model_comparison



# CREATE SPARK SESSION


spark = create_spark_session()



# LOAD DATA


df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv("../data/raw/online_retail.csv")
)

print("\nROWS BEFORE CLEANING:")
print(df.count())



# CLEAN DATA


df = clean_data(df)

print("\nROWS AFTER CLEANING:")
print(df.count())



# CREATE WEEKLY SALES DATASET


weekly_sales = create_weekly_sales(df)

print("\nWEEKLY SALES")
weekly_sales.show(20, truncate=False)

print("\nNUMBER OF WEEKS:")
print(weekly_sales.count())



# CREATE FORECASTING FEATURES


forecast_df = create_weekly_forecasting_dataset(weekly_sales)

print("\nWEEKLY FORECASTING DATASET")
forecast_df.show(20, truncate=False)



# BASELINE MODEL


baseline_predictions, baseline_mae, baseline_rmse, baseline_mape = (
    evaluate_baseline(forecast_df)
)

print("\nBASELINE METRICS")
print(f"MAE: {baseline_mae}")
print(f"RMSE: {baseline_rmse}")
print(f"MAPE: {baseline_mape}")



# LINEAR REGRESSION


lr_predictions, lr_mae, lr_rmse, lr_mape = (
    train_linear_regression_weekly(forecast_df)
)

print("\nLINEAR REGRESSION METRICS")
print(f"MAE: {lr_mae}")
print(f"RMSE: {lr_rmse}")
print(f"MAPE: {lr_mape}")



# RANDOM FOREST


rf_predictions, rf_mae, rf_rmse, rf_mape = (
    train_random_forest_weekly(forecast_df)
)

print("\nRANDOM FOREST METRICS")
print(f"MAE: {rf_mae}")
print(f"RMSE: {rf_rmse}")
print(f"MAPE: {rf_mape}")


# --------------------------------------------------
# GRADIENT BOOSTED TREES
# --------------------------------------------------

gbt_predictions, gbt_mae, gbt_rmse, gbt_mape = (
    train_gbt_weekly(forecast_df)
)

print("\nGRADIENT BOOSTED TREES METRICS")
print(f"MAE: {gbt_mae}")
print(f"RMSE: {gbt_rmse}")
print(f"MAPE: {gbt_mape}")



# MODEL COMPARISON


print("\nMODEL COMPARISON")
print("-" * 50)

print(f"Baseline MAE: {baseline_mae}")
print(f"Baseline RMSE: {baseline_rmse}")
print(f"Baseline MAPE: {baseline_mape}")

print()

print(f"Linear Regression MAE: {lr_mae}")
print(f"Linear Regression RMSE: {lr_rmse}")
print(f"Linear Regression MAPE: {lr_mape}")

print()

print(f"Random Forest MAE: {rf_mae}")
print(f"Random Forest RMSE: {rf_rmse}")
print(f"Random Forest MAPE: {rf_mape}")

print()

print(f"Gradient Boosted Trees MAE: {gbt_mae}")
print(f"Gradient Boosted Trees RMSE: {gbt_rmse}")
print(f"Gradient Boosted Trees MAPE: {gbt_mape}")



# SAVE RESULTS


comparison_df = save_model_comparison(
    spark,
    baseline_mae,
    baseline_rmse,
    baseline_mape,
    lr_mae,
    lr_rmse,
    lr_mape,
    rf_mae,
    rf_rmse,
    rf_mape,
    gbt_mae,
    gbt_rmse,
    gbt_mape
)

print("\nMODEL COMPARISON SAVED")

comparison_df.show(truncate=False)



# SHUTDOWN


spark.stop()
