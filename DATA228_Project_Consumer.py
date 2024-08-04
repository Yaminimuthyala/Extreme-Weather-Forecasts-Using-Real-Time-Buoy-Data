#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql.functions import col, from_json, expr
from pyspark.sql.functions import from_csv
from pyspark.sql.types import StructType, StructField, FloatType
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.ml.linalg import Vectors
import matplotlib.pyplot as plt
import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Initialize Spark session
spark = SparkSession.builder \
    .appName("KafkaConsumerApplication") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.0.1") \
    .getOrCreate()

# Define the schema of the input data
schema = StructType([
    StructField("WDIR", FloatType(), True),
    StructField("WSPD", FloatType(), True),
    StructField("GST", FloatType(), True),
    StructField("PRES", FloatType(), True),
    StructField("ATMP", FloatType(), True),
])

# Read the streaming data from Kafka
df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "test") \
    .load()

# # Load the trained model
model = RandomForestClassificationModel.load("models/random_forest_model")

# Select the 'value' column and cast it to a string
value_df = df.select(col("value").cast("string"))

# Deserialize JSON from the 'value' column using the provided schema
json_df = df.select(from_json(col("value").cast("string"), schema).alias("data"))

# Flatten the structure and select the JSON fields
flattened_df = json_df.select("data.*")

# UDF to extract the probability of the positive class
extract_prob_udf = udf(lambda v: float(v[1]), DoubleType())


def process_batch(batch_df, epoch_id):

    batch_df.show(truncate=False)
    
    # Define the assembler with the input column names that match your trained model's feature names
    assembler = VectorAssembler(inputCols=["WDIR", "WSPD", "GST", "PRES", "ATMP"], outputCol="features")

    # Transform the batch DataFrame to create the feature vector
    batch_df = assembler.transform(batch_df)

    # Ensure there is data in the batch before making predictions
    if batch_df.count() > 0:
        # Make predictions using the loaded model
        predictions = model.transform(batch_df)
        predictions = predictions.withColumn("probability", extract_prob_udf(col("probability")))

        # Select the relevant columns to display
        predictions = predictions.select(
            col("WDIR"),
            col("WSPD"),
            col("GST"),
            col("PRES"),
            col("ATMP"),
            col("probability"),
            col("prediction")
        )

        # Show the formatted predictions
        predictions.show(truncate=False)

# Apply the process_batch function to each micro-batch of data
query = flattened_df.writeStream.foreachBatch(process_batch).start()

query.awaitTermination()


# In[ ]:




