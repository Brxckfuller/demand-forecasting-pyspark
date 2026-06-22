from pyspark.sql import Row


def save_model_comparison(
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
):

    results = [
        Row(
            Model="Baseline",
            MAE=float(baseline_mae),
            RMSE=float(baseline_rmse),
            MAPE=float(baseline_mape)
        ),
        Row(
            Model="Linear Regression",
            MAE=float(lr_mae),
            RMSE=float(lr_rmse),
            MAPE=float(lr_mape)
        ),
        Row(
            Model="Random Forest",
            MAE=float(rf_mae),
            RMSE=float(rf_rmse),
            MAPE=float(rf_mape)
        ),
        Row(
            Model="Gradient Boosted Trees",
            MAE=float(gbt_mae),
            RMSE=float(gbt_rmse),
            MAPE=float(gbt_mape)
        )
    ]

    comparison_df = spark.createDataFrame(results)

    (
        comparison_df
        .coalesce(1)
        .write
        .mode("overwrite")
        .option("header", True)
        .csv("../data/outputs/model_comparison")
    )

    return comparison_df