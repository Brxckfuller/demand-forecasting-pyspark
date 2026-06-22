from pyspark.sql import functions as F


def clean_data(df):

    # Convert InvoiceDate from string to timestamp
    df = df.withColumn(
        "InvoiceDate",
        F.to_timestamp("InvoiceDate", "d/M/yyyy H:mm")
    )

    # Remove rows with missing product codes
    df = df.filter(F.col("StockCode").isNotNull())

    # Remove rows with missing dates
    df = df.filter(F.col("InvoiceDate").isNotNull())

    # Remove rows with missing quantities
    df = df.filter(F.col("Quantity").isNotNull())

    # Remove returns/cancellations
    df = df.filter(F.col("Quantity") > 0)

    # Remove invalid prices
    df = df.filter(F.col("UnitPrice") > 0)

    return df