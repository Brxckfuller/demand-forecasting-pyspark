from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator


FEATURE_COLUMNS = [
    "Lag_1_Week",
    "Lag_2_Week",
    "Lag_3_Week",
    "Lag_4_Week",
    "Rolling_4_Week_Avg"
]


def prepare_data(df):
    assembler = VectorAssembler(
        inputCols=FEATURE_COLUMNS,
        outputCol="features"
    )

    data = assembler.transform(df)

    train_data = data.filter(F.col("Week") <= 39)
    test_data = data.filter(F.col("Week") > 39)

    return train_data, test_data


def get_metrics(predictions):
    mae = RegressionEvaluator(
        labelCol="TotalSales",
        predictionCol="prediction",
        metricName="mae"
    ).evaluate(predictions)

    rmse = RegressionEvaluator(
        labelCol="TotalSales",
        predictionCol="prediction",
        metricName="rmse"
    ).evaluate(predictions)

    mape = (
        predictions
        .withColumn(
            "APE",
            F.abs((F.col("TotalSales") - F.col("prediction")) / F.col("TotalSales")) * 100
        )
        .agg(F.avg("APE").alias("MAPE"))
        .collect()[0]["MAPE"]
    )

    return mae, rmse, mape


def evaluate_baseline(df):
    predictions = df.withColumn(
        "prediction",
        F.col("Lag_1_Week").cast("double")
    )

    mae, rmse, mape = get_metrics(predictions)

    return predictions, mae, rmse, mape


def train_linear_regression_weekly(df):
    train_data, test_data = prepare_data(df)

    model = LinearRegression(
        featuresCol="features",
        labelCol="TotalSales"
    ).fit(train_data)

    predictions = model.transform(test_data)

    mae, rmse, mape = get_metrics(predictions)

    return predictions, mae, rmse, mape


def train_random_forest_weekly(df):
    train_data, test_data = prepare_data(df)

    model = RandomForestRegressor(
        featuresCol="features",
        labelCol="TotalSales",
        numTrees=100,
        maxDepth=5,
        seed=42
    ).fit(train_data)

    predictions = model.transform(test_data)

    mae, rmse, mape = get_metrics(predictions)

    return predictions, mae, rmse, mape


def train_gbt_weekly(df):
    train_data, test_data = prepare_data(df)

    model = GBTRegressor(
        featuresCol="features",
        labelCol="TotalSales",
        maxIter=100,
        maxDepth=4,
        seed=42
    ).fit(train_data)

    predictions = model.transform(test_data)

    mae, rmse, mape = get_metrics(predictions)

    return predictions, mae, rmse, mape