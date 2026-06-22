from pyspark.sql import functions as F


def evaluate_baseline_model(forecast_df):
    results = (
        forecast_df
        .withColumn("Prediction", F.col("Lag_1_Month"))
        .withColumn("Error", F.col("TotalSales") - F.col("Prediction"))
        .withColumn("AbsoluteError", F.abs(F.col("Error")))
        .withColumn("SquaredError", F.col("Error") ** 2)
        .withColumn(
            "AbsolutePercentageError",
            F.abs(F.col("Error") / F.col("TotalSales")) * 100
        )
    )

    metrics = (
        results
        .agg(
            F.avg("AbsoluteError").alias("MAE"),
            F.sqrt(F.avg("SquaredError")).alias("RMSE"),
            F.avg("AbsolutePercentageError").alias("MAPE")
        )
    )

    return results, metrics