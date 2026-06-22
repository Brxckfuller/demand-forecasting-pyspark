from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator


def train_linear_regression(df):

    feature_columns = [
        "Lag_1_Month",
        "Lag_2_Month",
        "Lag_3_Month",
        "Rolling_3_Month_Avg"
    ]

    assembler = VectorAssembler(
        inputCols=feature_columns,
        outputCol="features"
    )

    data = assembler.transform(df)

    train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

    lr = LinearRegression(
        featuresCol="features",
        labelCol="TotalSales"
    )

    model = lr.fit(train_data)

    predictions = model.transform(test_data)

    rmse = RegressionEvaluator(
        labelCol="TotalSales",
        predictionCol="prediction",
        metricName="rmse"
    ).evaluate(predictions)

    mae = RegressionEvaluator(
        labelCol="TotalSales",
        predictionCol="prediction",
        metricName="mae"
    ).evaluate(predictions)

    return predictions, mae, rmse