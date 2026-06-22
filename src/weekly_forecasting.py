from pyspark.sql import functions as F
from pyspark.sql.window import Window


def create_weekly_forecasting_dataset(weekly_sales):
    window_spec = Window.orderBy("Year", "Week")

    forecast_df = (
        weekly_sales
        .withColumn("Lag_1_Week", F.lag("TotalSales", 1).over(window_spec))
        .withColumn("Lag_2_Week", F.lag("TotalSales", 2).over(window_spec))
        .withColumn("Lag_3_Week", F.lag("TotalSales", 3).over(window_spec))
        .withColumn("Lag_4_Week", F.lag("TotalSales", 4).over(window_spec))
        .withColumn(
            "Rolling_4_Week_Avg",
            F.avg("TotalSales").over(window_spec.rowsBetween(-4, -1))
        )
        .dropna()
    )

    return forecast_df