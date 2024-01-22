import streamlit as st
import pickle
import pandas as pd
import numpy as np
import pathlib
import boto3
import yaml
from io import StringIO

def load_data(bucket, key, aws_access_key_id=None, aws_secret_access_key=None, region_name=None):
    """
    Load dataset from an S3 bucket.

    Parameters:
    - bucket (str): S3 bucket name.
    - key (str): S3 object key (path to the file within the bucket).
    - aws_access_key_id (str, optional): AWS access key ID. Defaults to None.
    - aws_secret_access_key (str, optional): AWS secret access key. Defaults to None.
    - region_name (str, optional): AWS region name. Defaults to None.

    Returns:
    - pd.DataFrame: Loaded DataFrame from the S3 object.
    """
    # Initialize S3 client
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=region_name)
    
    # Get S3 object
    obj = s3.get_object(Bucket=bucket, Key=key)
    
    # Read CSV data from S3 object's body
    df = pd.read_csv(obj['Body'])
    
    return df

def load_pickled_model(bucket, key, aws_access_key_id=None, aws_secret_access_key=None, region_name=None):
    """
    Load a pickled model from an S3 bucket.

    Parameters:
    - bucket (str): S3 bucket name.
    - key (str): S3 object key (path to the file within the bucket).
    - aws_access_key_id (str, optional): AWS access key ID. Defaults to None.
    - aws_secret_access_key (str, optional): AWS secret access key. Defaults to None.
    - region_name (str, optional): AWS region name. Defaults to None.

    Returns:
    - Loaded pickled model or data
    """
    # Initialize S3 client
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=region_name)

    # Download pickled model from S3
    response = s3.get_object(Bucket=bucket, Key=key)
    pickled_model = response['Body'].read()

    # Load pickled model
    loaded_model = pickle.loads(pickled_model)

    return loaded_model

# Set Streamlit page configuration
st.set_page_config(page_title="Price Predictor")

# Get the current directory path
curr_dir = pathlib.Path(__file__)

# Navigate up three levels to reach the parent directory
home_dir = curr_dir.parent.parent

# Define the path to the 'params.yaml' file within the home directory
params_file = home_dir.as_posix() + '/params.yaml'

# Load S3 parameters from 'params.yaml'
s3_params = yaml.safe_load(open(params_file))["s3"]

# Load file-specific parameters for 'data-preprocessing-flats' from 'params.yaml'
file_params = yaml.safe_load(open(params_file))["run-streamlit"]

# Extract S3 parameters from the loaded 's3_params'
s3_bucket = s3_params['bucket']
s3_key = file_params['price_predictor_data']
model_s3_key = file_params['train_model']
aws_access_key_id = s3_params['aws_access_key_id']
aws_secret_access_key = s3_params['aws_secret_access_key']
region_name = s3_params['region_name']

# # Load the DataFrame and pipeline model from pickle files
# df = pd.read_csv(home_dir + '/data/processed/gurgaon_properties_post_feature_selection.csv')

# Load data from S3 using specified parameters
df = load_data(bucket=s3_bucket,
                    key=s3_key,
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=region_name)

# with open(home_dir + '/models/trained_model.pkl', 'rb') as file:
#     pipeline = pickle.load(file)

# Load the saved model
pipeline = load_pickled_model(bucket=s3_bucket,
        key=model_s3_key,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name)

# Streamlit app header
st.header('Enter your inputs')

# Property Type selection
property_type = st.selectbox('Property Type', ['flat', 'house'])

# Sector selection
sector = st.selectbox('Sector', sorted(df['sector'].unique().tolist()))

# Number of Bedrooms selection
bedrooms = float(st.selectbox('Number of Bedrooms', sorted(df['bedRoom'].unique().tolist())))

# Number of Bathrooms selection
bathroom = float(st.selectbox('Number of Bathrooms', sorted(df['bathroom'].unique().tolist())))

# Balconies selection
balcony = st.selectbox('Balconies', sorted(df['balcony'].unique().tolist()))

# Property Age selection
property_age = st.selectbox('Property Age', sorted(df['agePossession'].unique().tolist()))

# Built Up Area input
built_up_area = float(st.number_input('Built Up Area'))

# Servant Room selection
servant_room = float(st.selectbox('Servant Room', [0.0, 1.0]))

# Store Room selection
store_room = float(st.selectbox('Store Room', [0.0, 1.0]))

# Furnishing Type selection
furnishing_type = st.selectbox('Furnishing Type', sorted(df['furnishing_type'].unique().tolist()))

# Luxury Category selection
luxury_category = st.selectbox('Luxury Category', sorted(df['luxury_category'].unique().tolist()))

# Floor Category selection
floor_category = st.selectbox('Floor Category', sorted(df['floor_category'].unique().tolist()))

# Prediction button
if st.button('Predict'):

    # Form a DataFrame from user inputs
    data = [[property_type, sector, bedrooms, bathroom, balcony, property_age, built_up_area, servant_room, store_room, furnishing_type, luxury_category, floor_category]]
    columns = ['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
               'agePossession', 'built_up_area', 'servant room', 'store room',
               'furnishing_type', 'luxury_category', 'floor_category']

    # Convert to DataFrame
    one_df = pd.DataFrame(data, columns=columns)

    # Predict using the pipeline model
    base_price = np.expm1(pipeline.predict(one_df))[0]
    low = base_price - 0.22
    high = base_price + 0.22

    # Display the predicted price range
    st.text("The price of the flat is between {} Cr and {} Cr".format(round(low, 2), round(high, 2)))
