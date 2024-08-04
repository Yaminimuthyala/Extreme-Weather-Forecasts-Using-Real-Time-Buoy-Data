# Extreme Weather Forecasts Using Real-Time Buoy Data
### Overview
This project focuses on deploying a big data system to enhance the forecasting of large storms and weather patterns along the California coast. The system aims to improve forecast accuracy and deliver timely alerts for dangerous phenomena such as enormous waves, typhoons, and hurricanes. By leveraging modern machine learning algorithms and big data analytics, this project demonstrates the potential of integrating machine learning with comprehensive environmental datasets to improve the forecasting of coastal weather phenomena.

### Key Features
### Data Sources:
Utilizes extensive buoy data from the National Data Buoy Center (NDBC), including wave patterns, wind speeds, air pressure, and water temperature.
### Machine Learning Models: 
Implements and compares the performance of various models such as Random Forest, Gradient Boosting, and XGBoost.
### Real-Time Processing: 
Integrates Apache Kafka and Spark Streaming for real-time data processing and prediction.
### Predictive Accuracy: 
Achieved accuracy percentages of 92.01% for Random Forest, 88.43% for Gradient Boosting, and 89.61% for XGBoost.

## Methodology
### Data Collection:

Historical data from NDBC buoy stations (2014-2023).
Storm events data from NOAA's Storm Events Database.
### Data Processing:

Handling missing values and merging buoy data with storm events data.
Exploratory data analysis to identify trends and anomalies.
### Model Development:

Training and evaluating models using Random Forest, Gradient Boosting, and XGBoost.
Hyperparameter tuning to optimize model performance.
### Real-Time Implementation:

Using Kafka for real-time data ingestion.
Spark Streaming for applying predictive models to live data.
### Results
The Random Forest model achieved the highest accuracy, recall, and F1 score, making it the best-performing model for this project.
The system successfully provided timely and accurate predictions for severe weather events when applied to real-time data streams.

### DATA228_Project_EDA.ipynb: Exploratory data analysis process.
### DATA228_Project_ML_Models.ipynb: Historical data preprocessing and machine learning model development.
### DATA228_Project_ML_Models_for_local_use.py: Converts machine learning models for local use.
### DATA228_Project_Producer.ipynb: Kafka producer for real-time data ingestion.
### DATA228_Project_Consumer.ipynb: Kafka consumer for real-time data processing and prediction.
