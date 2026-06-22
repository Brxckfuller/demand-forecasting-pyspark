from pyspark.sql.functions import (
    year,
    weekofyear,
    sum
)


def create_weekly_sales(df):

    weekly_sales = (
        df
        .withColumn(
            "Year",
            year("InvoiceDate")
        )
        .withColumn(
            "Week",
            weekofyear("InvoiceDate")
        )
        .groupBy(
            "Year",
            "Week"
        )
        .agg(
            sum("Quantity").alias("TotalSales")
        )
        .orderBy(
            "Year",
            "Week"
        )
    )

    return weekly_sales 