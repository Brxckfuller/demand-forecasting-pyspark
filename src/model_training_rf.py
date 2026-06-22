from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator


def train_random_forest(df):

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

    train_data, test_data = data.randomSplit(
        [0.8, 0.2],
        seed=42
    )

    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="TotalSales",
        numTrees=100,
        maxDepth=5,
        seed=42
    )

    model = rf.fit(train_data)

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