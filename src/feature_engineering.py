from pyspark.sql.functions import to_timestamp
from pyspark.sql.functions import year
from pyspark.sql.functions import month
from pyspark.sql.functions import sum


def create_monthly_sales(df):

    df = df.withColumn(
        "InvoiceDate",
        to_timestamp("InvoiceDate", "d/M/yyyy H:mm")
    )

    monthly_sales = (
        df.groupBy(
            year("InvoiceDate").alias("Year"),
            month("InvoiceDate").alias("Month")
        )
        .agg(
            sum("Quantity").alias("TotalSales")
        )
        .orderBy("Year", "Month")
    )

    return monthly_sales