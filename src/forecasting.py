from pyspark.sql import functions as F
from pyspark.sql.window import Window


def create_forecasting_dataset(monthly_sales):
    window_spec = Window.orderBy("Year", "Month")

    forecast_df = (
        monthly_sales
        .withColumn("Lag_1_Month", F.lag("TotalSales", 1).over(window_spec))
        .withColumn("Lag_2_Month", F.lag("TotalSales", 2).over(window_spec))
        .withColumn("Lag_3_Month", F.lag("TotalSales", 3).over(window_spec))
        .withColumn(
            "Rolling_3_Month_Avg",
            F.avg("TotalSales").over(
                window_spec.rowsBetween(-3, -1)
            )
        )
        .dropna()
    )

    return forecast_df 