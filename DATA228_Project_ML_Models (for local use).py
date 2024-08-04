#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer
import os
import sys

np.random.seed(42)

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


# Load feature data

# In[2]:


import pandas as pd
import numpy as np

# Set the dataset path to local folder
dataset_path = 'dataset/'

# List of all feature file paths
feature_file_paths = [f'{dataset_path}46013h{year}.txt' for year in range(2014, 2023)] + \
                     [f'{dataset_path}46026h{year}.txt' for year in range(2014, 2023)] + \
                     [f'{dataset_path}46237h{year}.txt' for year in range(2014, 2023)] + \
                     [f'{dataset_path}ftpc1h{year}.txt' for year in range(2014, 2023)] + \
                     [f'{dataset_path}pxoc1h{year}.txt' for year in range(2014, 2023)] + \
                     [f'{dataset_path}pxsc1h{year}.txt' for year in range(2014, 2023)] + \
                     [f'{dataset_path}tibc1h{year}.txt' for year in range(2015, 2023)]

# Initialize an empty DataFrame to store the combined feature data
all_feature_data = pd.DataFrame()

# Load and concatenate all feature files
for feature_file_path in feature_file_paths:
    feature_data = pd.read_csv(feature_file_path, delim_whitespace=True, skiprows=[1])
    feature_data['timestamp'] = pd.to_datetime(feature_data[['#YY', 'MM', 'DD', 'hh', 'mm']].astype(str).agg(' '.join, axis=1), format='%Y %m %d %H %M')
    feature_data['year'] = feature_data['timestamp'].dt.year
    feature_data['month'] = feature_data['timestamp'].dt.month
    feature_data['day'] = feature_data['timestamp'].dt.day
    feature_data['hour'] = feature_data['timestamp'].dt.hour
    feature_data['minute'] = feature_data['timestamp'].dt.minute
    all_feature_data = pd.concat([all_feature_data, feature_data], axis=0, ignore_index=True)

print(f'total count of feature data: {all_feature_data.shape[0]}')

# Define the missing value patterns
missing_patterns = [99.00, 999, 999.0, 99.0]

# Loop through each column and replace each pattern with NaN
for column in all_feature_data.columns:
    for pattern in missing_patterns:
        all_feature_data[column] = all_feature_data[column].replace(to_replace=pattern, value=np.nan, regex=True)

missing_data = all_feature_data.isnull().sum()

# Display missing data count for each column
print('missing data count in raw data:')
print(missing_data)

# Setting threshold for excessive missing values (e.g., 60%)
threshold = 0.6 * len(all_feature_data)

# Drop columns with missing values greater than the threshold
all_feature_data.dropna(axis=1, thresh=threshold, inplace=True)

# Drop rows with any missing values
all_feature_data.dropna(axis=0, inplace=True)


# Load target file

# In[3]:


# Load target file
target_file_path = 'dataset/storm_data_search_results.csv'
target_data = pd.read_csv(target_file_path, sep=',')

# Keep only specific columns
selected_columns = ['BEGIN_DATE', 'BEGIN_TIME', 'EVENT_TYPE']
target_data = target_data[selected_columns]

# Convert the TIME columns to hourly timestamp
target_data['BEGIN_TIME'] = target_data['BEGIN_TIME'].floordiv(100).astype(str).str.pad(2, fillchar='0') + '00'

# Convert timestamp columns to a single datetime column
target_data['timestamp'] = pd.to_datetime(target_data[['BEGIN_DATE', 'BEGIN_TIME']].astype(str).agg(' '.join, axis=1), format='%m/%d/%Y %H%M')

# Display the first few rows to verify correct loading and formatting
print(target_data.head())


# In[4]:


'''Merge feature and target data based on the timestamp'''
# Merge feature and target data based on the timestamp as event data
all_event_data = pd.merge(all_feature_data, target_data, how='right', on='timestamp')

# Merge feature and target data based on the timestamp as other data
all_other_data = pd.merge(all_feature_data, target_data, how='left', on='timestamp')
all_other_data['EVENT_TYPE'].fillna('no', inplace=True)

# concatenate two partial data into a whole dataset
all_data = pd.concat([all_event_data, all_other_data])

# drop some unuseful columns
all_data = all_data.drop(['timestamp','BEGIN_DATE', 'BEGIN_TIME', '#YY', 'MM', 'DD', 'hh', 'mm', 'year', 'month', 'day', 'hour', 'minute'], axis=1)
# Drop rows with any missing values
all_data.dropna(axis=0, inplace=True)
all_data.loc[all_data["EVENT_TYPE"] != "no", "EVENT_TYPE"] = 'yes'

print('missing data count in prepared data:')
missing_data_prepared = all_data.isnull().sum()
print(missing_data_prepared)

print(f'total count of prepared data: {len(all_data)}')


# In[5]:


all_data


# Identify features and target variable

# In[6]:


# Identify features and target variable
X = all_data.drop(['EVENT_TYPE'], axis=1)
y = all_data['EVENT_TYPE']

yes_count = all_data['EVENT_TYPE'].value_counts().get('yes', 0)
print(yes_count)


# Split training and test dataset

# In[7]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
train_data = pd.concat([X_train, y_train], axis=1)

# Identify the minority class label
minority_class_label = train_data['EVENT_TYPE'].value_counts().idxmin()

# Apply random undersampling
undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(train_data.drop('EVENT_TYPE', axis=1), train_data['EVENT_TYPE'])
print(y_resampled.value_counts())

print(type(X_resampled))

df_train = pd.concat([X_resampled, y_resampled], axis = 1)
df_test = pd.concat([X_test, y_test], axis = 1)


# In[8]:


spark = SparkSession.builder.getOrCreate()

# create DataFrame
df_train_spark = spark.createDataFrame(df_train)

df_test_spark = spark.createDataFrame(df_test)


# In[9]:


# Create a feature vector by combining all the features
assembler = VectorAssembler(inputCols=["WDIR", "WSPD", "GST", "PRES", "ATMP"], outputCol="features")

# Transform the data to create the feature vector
train_data = assembler.transform(df_train_spark)
test_data = assembler.transform(df_test_spark)

label_stringIdx = StringIndexer(inputCol = 'EVENT_TYPE', outputCol = 'labelIndex')
train_data = label_stringIdx.fit(train_data).transform(train_data)
test_data = label_stringIdx.fit(test_data).transform(test_data)
train_data.show()


# Random Forest Classifier

# In[10]:


from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'labelIndex', numTrees=40, maxDepth=3)
rf_model = rf.fit(train_data)
rf_predictions = rf_model.transform(test_data)
# rf_predictions.select('labelIndex', 'rawPrediction', 'prediction', 'probability').show(25)


# In[11]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator

rf_evaluator = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction")
accuracy = rf_evaluator.evaluate(rf_predictions)
print(f'Accuracy = {accuracy:.4f}')
print(f'Test Error = {(1.0 - accuracy):.4f}')


# Gradient Boosting Classifier

# In[12]:


from pyspark.ml.classification import GBTClassifier

gb = GBTClassifier(featuresCol = 'features', labelCol = 'labelIndex', maxIter=50)
gb_model = gb.fit(train_data)
gb_predictions = gb_model.transform(test_data)


# In[13]:


gb_evaluator = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction")
accuracy = gb_evaluator.evaluate(gb_predictions)
print(f'Accuracy = {accuracy:.4f}')
print(f'Test Error = {(1.0 - accuracy):.4f}')


# XGBoost

# In[14]:


# pip install xgboost
# !pip install pyarrow


# In[15]:


from xgboost.spark import SparkXGBClassifier

xgb = SparkXGBClassifier(label_col="labelIndex", missing=0.0)
xgb_model = xgb.fit(train_data)
xgb_predictions = xgb_model.transform(test_data)


# In[16]:


xgb_evaluator = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction")
accuracy = xgb_evaluator.evaluate(xgb_predictions)
print(f'Accuracy = {accuracy:.4f}')
print(f'Test Error = {(1.0 - accuracy):.4f}')


# In[18]:


rf_model.save("models/random_forest_model")
gb_model.save("models/gradient_boosted_trees_model")
xgb_model.write().overwrite().save("models/xgboost_model")


# In[ ]:




