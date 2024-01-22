# data-preprocessing-houses.py
import pathlib
import sys
import pandas as pd
import numpy as np
import re
import boto3
import yaml
from io import StringIO

# def load_data(data_path):
#     # Load your dataset from a given path
#     df = pd.read_csv(data_path)
#     print(df.head())
#     return df

# def save_data(data, output_path):
#     # Save the split datasets to the specified output path
#     pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
#     data.to_csv(output_path + '/house_cleaned.csv', index=False)

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

def save_data(data, bucket, key, aws_access_key_id=None, aws_secret_access_key=None, region_name=None):
    """
    Save dataset to an S3 bucket.

    Parameters:
    - data (pd.DataFrame): DataFrame to be saved.
    - bucket (str): S3 bucket name.
    - key (str): S3 object key (path to the file within the bucket).
    - aws_access_key_id (str, optional): AWS access key ID. Defaults to None.
    - aws_secret_access_key (str, optional): AWS secret access key. Defaults to None.
    - region_name (str, optional): AWS region name. Defaults to None.

    Returns:
    - None
    """
    # Initialize S3 client
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=region_name)
    
    # Convert DataFrame to CSV format in memory
    csv_buffer = StringIO()
    data.to_csv(csv_buffer, index=False)
    
    # Upload CSV data to S3 object
    s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())

def clean_society_col(df):
    """
    Clean the 'society' column in the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with the 'society' column cleaned.
    """
    # Remove digits and special characters from 'society' names, convert to lowercase
    df['society'] = df['society'].apply(lambda name: re.sub(r'\d+(\.\d+)?\s?★', '', str(name)).strip()).str.lower()
    
    # Replace 'nan' values with 'independent'
    df['society'] = df['society'].str.replace('nan', 'independent')

    return df

def drop_and_rename_cols(df):
    """
    Drop and rename columns in the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with specified columns dropped and renamed.
    """
    # Drop 'link' and 'property_id' columns
    df.drop(columns=['link', 'property_id'], inplace=True)
    
    # Rename 'rate' column to 'price_per_sqft'
    df.rename(columns={'rate': 'price_per_sqft'}, inplace=True)

    return df

def process_price_column(df):
    """
    Process the 'price' column in the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with processed 'price' column.
    """
    # Filter out rows with 'Price on Request'
    df = df[df['price'] != 'Price on Request']

    def treat_price(x):
        if type(x) == float:
            return x
        else:
            if x[1] == 'Lac':
                return round(float(x[0]) / 100, 2)
            else:
                return round(float(x[0]), 2)

    # Apply the 'treat_price' function to the 'price' column
    df['price'] = df['price'].str.split(' ').apply(treat_price)

    return df

def clean_price_per_sqft_col(df):
    """
    Clean the 'price_per_sqft' column in the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with cleaned 'price_per_sqft' column.
    """
    # Clean the 'price_per_sqft' column
    df['price_per_sqft'] = df['price_per_sqft'].str.split('/').str.get(0).str.replace('₹','').str.replace(',','').str.strip().astype('float')

    return df

def treat_bedroom_col(df):
    """
    Treat the 'bedRoom' column in the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with treated 'bedRoom' column.
    """
    # Drop rows where 'bedRoom' is null
    df = df[~df['bedRoom'].isnull()]

    # Extract numeric part from 'bedRoom' and convert to int
    df['bedRoom'] = df['bedRoom'].str.split(' ').str.get(0).astype('int')

    return df

def treat_bathroom_col(df):
    """
    Treat the 'bathroom' column in the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with treated 'bathroom' column.
    """
    # Extract numeric part from 'bathroom' and convert to int
    df['bathroom'] = df['bathroom'].str.split(' ').str.get(0).astype('int')

    return df

def treat_balcony_col(df):
    """
    Treat the 'balcony' column in the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with treated 'balcony' column.
    """
    # Extract numeric part from 'balcony' and replace 'No' with '0'
    df['balcony'] = df['balcony'].str.split(' ').str.get(0).str.replace('No','0')

    return df

def treat_additional_room_col(df):
    """
    Treat the 'additionalRoom' column in the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with treated 'additionalRoom' column.
    """
    # Fill NaN values with 'not available' and convert to lowercase
    df['additionalRoom'].fillna('not available', inplace=True)
    df['additionalRoom'] = df['additionalRoom'].str.lower()

    return df

def treat_noOfFloor_col(df):
    """
    Treat the 'noOfFloor' column in the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with treated 'noOfFloor' column.
    """
    # Extract the numeric part and rename the column to 'floorNum'
    df['noOfFloor'] = df['noOfFloor'].str.split(' ').str.get(0)
    df.rename(columns={'noOfFloor':'floorNum'}, inplace=True)

    return df

def treat_facing_col(df):
    """
    Treat the 'facing' column in the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with treated 'facing' column.
    """
    # Fill missing values with 'NA'
    df['facing'].fillna('NA', inplace=True)

    return df

def calculate_area(df):
    """
    Calculate the 'area' column in the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with calculated 'area' column.
    """
    df['area'] = round((df['price'] * 10000000) / df['price_per_sqft'])
    return df


def main():

    # Get the current directory path
    curr_dir = pathlib.Path(__file__)

    # Navigate up three levels to reach the parent directory
    home_dir = curr_dir.parent.parent.parent

    # Define the path to the 'params.yaml' file within the home directory
    params_file = home_dir.as_posix() + '/params.yaml'

    # Load S3 parameters from 'params.yaml'
    s3_params = yaml.safe_load(open(params_file))["s3"]

    # Load file-specific parameters for 'data-preprocessing-houses' from 'params.yaml'
    file_params = yaml.safe_load(open(params_file))["data-preprocessing-houses"]

    # curr_dir = pathlib.Path(__file__)
    # home_dir = curr_dir.parent.parent.parent

    # input_file = sys.argv[1]
    # data_path = home_dir.as_posix() + input_file
    # output_path = home_dir.as_posix() + '/data/processed'

    # Extract S3 parameters from the loaded 's3_params'
    s3_bucket = s3_params['bucket']
    s3_key = file_params['input_data']
    output_s3_key = file_params['output_data']
    aws_access_key_id = s3_params['aws_access_key_id']
    aws_secret_access_key = s3_params['aws_secret_access_key']
    region_name = s3_params['region_name']
    
    # Load data from S3 using specified parameters
    data = load_data(bucket=s3_bucket,
                     key=s3_key,
                     aws_access_key_id=aws_access_key_id,
                     aws_secret_access_key=aws_secret_access_key,
                     region_name=region_name)
    
    # Apply data preprocessing functions
    data = data.drop_duplicates()
    data = drop_and_rename_cols(data)
    data = clean_society_col(data)
    data = process_price_column(data)
    data = clean_price_per_sqft_col(data)
    data = treat_bedroom_col(data)
    data = treat_bathroom_col(data)
    data = treat_balcony_col(data)
    data = treat_additional_room_col(data)
    data = treat_noOfFloor_col(data)
    data = treat_facing_col(data)
    data = calculate_area(data)

    # Insert 'property_type' column with value 'house'
    data.insert(loc=1,column='property_type',value='house')
    
    # Save the processed data back to S3
    save_data(data=data,
              bucket=s3_bucket,
              key=output_s3_key,
              aws_access_key_id=aws_access_key_id,
              aws_secret_access_key=aws_secret_access_key,
              region_name=region_name)

if __name__ == "__main__":
    main()