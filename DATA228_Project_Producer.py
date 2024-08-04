#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from kafka import KafkaProducer
import requests
import pandas as pd
from io import StringIO
import time
import numpy as np

# Kafka Producer Configuration
producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda x: x.encode('utf-8'))

# Topic to send messages to
topic = 'test'

# URL of the data source
url = "https://www.ndbc.noaa.gov/data/latest_obs/latest_obs.txt"

# List of headers for the DataFrame
headers = [
    'STN', 'LAT', 'LON', 'YYYY', 'MM', 'DD', 'hh', 'mm', 'WDIR', 
    'WSPD', 'GST', 'WVHT', 'DPD', 'APD', 
    'MWD', 'PRES', 'PTDY', 'ATMP', 
    'WTMP', 'DEWP', 'VIS', 'TIDE'
]

# List of station IDs to filter by
station_ids = [
    '46011', '46013', '46014', '46022', '46025', '46026', '46027', 
    '46028', '46042', '46053', '46054', '46069', '46086'
]

def fetch_and_clean_data(url, headers, station_ids):
    # Fetch the data
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code != 200:
        print(f"Error fetching data: {response.status_code}")
        return None

    # Read the content of the response into a Pandas DataFrame
    data = pd.read_csv(StringIO(response.text), delim_whitespace=True, comment='#', names=headers, header=None)

    # Convert the 'STN' to string and filter based on the station IDs list
    data['STN'] = data['STN'].astype(str)
    data = data[data['STN'].isin(station_ids)]
    
    data.replace('MM', np.nan, inplace=True)

    # Ensure columns that should be numeric are of numeric type
    for col in ["WDIR", "WSPD", "GST", "PRES", "ATMP"]:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Impute missing values with the median of each column
    for column in data.columns:
        # Skip non-numeric columns
        if data[column].dtype == float or data[column].dtype == int:
            median = data[column].median()
            if pd.notnull(median):  # Check if median is not NaN
                data[column].fillna(median, inplace=True)

    # Drop the original date and time columns
    data.drop(['STN', 'LAT', 'LON', 'YYYY', 'MM', 'DD', 'hh', 'mm', 
               'WVHT', 'DPD', 'APD', 'MWD', 'PTDY', 'WTMP', 'DEWP', 'VIS', 'TIDE'], axis=1, inplace=True)

    # Convert the cleaned DataFrame to a JSON string to send to Kafka
    return data.to_json(orient='records', lines=True)

# Producer loop
try:
    while True:
        clean_data = fetch_and_clean_data(url, headers, station_ids)
        if clean_data:
            for record in clean_data.splitlines():
                producer.send(topic, value=record)
            producer.flush()  # Ensure data is sent to Kafka promptly
            print(f"Sent cleaned data to topic '{topic}'")
        else:
            print("No data to send or there was an error fetching the data.")
        time.sleep(300)  # Wait for 5 minutes before fetching again
except KeyboardInterrupt:
    print("Stopped fetching data.")
finally:
    producer.close()


# In[ ]:




