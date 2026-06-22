def save_monthly_sales(df):

    (
        df.coalesce(1)
          .write
          .mode("overwrite")
          .option("header", True)
          .csv("../data/processed/monthly_sales")
    )

