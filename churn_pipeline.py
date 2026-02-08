from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 1. Инициализация Spark
spark = SparkSession.builder.appName("CustomerChurnComparison").getOrCreate()

# 2. Загрузка данных
data = spark.read.csv("hdfs:///user/hadoop/churn_input/Churn_Modelling.csv", header=True, inferSchema=True)

# Очистка данных (удаление ненужных столбцов)
data = data.drop("RowNumber", "CustomerId", "Surname")

# Разделение на обучение и тест
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# 3. Feature Engineering Stages
geo_indexer = StringIndexer(inputCol="Geography", outputCol="GeographyIndex")
gender_indexer = StringIndexer(inputCol="Gender", outputCol="GenderIndex")

encoder = OneHotEncoder(
    inputCols=["GeographyIndex", "GenderIndex"], 
    outputCols=["GeographyVec", "GenderVec"]
)

assembler = VectorAssembler(
    inputCols=["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary", "GeographyVec", "GenderVec"],
    outputCol="features"
)

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

# 4. Модели для эксперимента (Option C)
lr = LogisticRegression(labelCol="Exited", featuresCol="scaledFeatures")
rf = RandomForestClassifier(labelCol="Exited", featuresCol="scaledFeatures", numTrees=50)

# 5. Pipeline для Logistic Regression
pipeline_lr = Pipeline(stages=[geo_indexer, gender_indexer, encoder, assembler, scaler, lr])
model_lr = pipeline_lr.fit(train_data)
predictions_lr = model_lr.transform(test_data)

# 6. Pipeline для Random Forest
pipeline_rf = Pipeline(stages=[geo_indexer, gender_indexer, encoder, assembler, scaler, rf])
model_rf = pipeline_rf.fit(train_data)
predictions_rf = model_rf.transform(test_data)

# 7. Оценка
evaluator = MulticlassClassificationEvaluator(labelCol="Exited", predictionCol="prediction", metricName="accuracy")

acc_lr = evaluator.evaluate(predictions_lr)
acc_rf = evaluator.evaluate(predictions_rf)

print("\n" + "="*30)
print(f"Logistic Regression Accuracy: {acc_lr:.4f}")
print(f"Random Forest Accuracy: {acc_rf:.4f}")
print("="*30 + "\n")

spark.stop()