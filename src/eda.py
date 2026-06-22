from pyspark.sql import functions as F


def run_eda(df):

    print("\nTOP 10 PRODUCTS")

    (
        df.groupBy("Description")
        .agg(F.sum("Quantity").alias("TotalQuantity"))
        .orderBy(F.desc("TotalQuantity"))
        .show(10, truncate=False)
    )

    print("\nTOP 10 COUNTRIES")

    (
        df.groupBy("Country")
        .agg(F.sum("Quantity").alias("TotalQuantity"))
        .orderBy(F.desc("TotalQuantity"))
        .show(10, truncate=False)
    )