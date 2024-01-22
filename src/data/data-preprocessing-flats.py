# data-preprocessing-flats.py
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
#     return df

# def save_data(data, output_path):
#     pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
#     data.to_csv(output_path + '/flats_cleaned.csv', index=False)

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

def drop_and_rename_cols(df):
    """
    Drop specified columns and rename 'area' to 'price_per_sqft' in the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with specified columns dropped and renamed.
    """
    # Drop columns 'link' and 'property_id'
    df.drop(columns=['link', 'property_id'], inplace=True)
    
    # Rename 'area' column to 'price_per_sqft'
    df.rename(columns={'area': 'price_per_sqft'}, inplace=True)
    
    return df

def clean_society_col(df):
    """
    Clean the 'society' column in the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with 'society' column cleaned.
    """
    # Apply a lambda function to remove digits and special characters from 'society' values
    df['society'] = df['society'].apply(lambda name: re.sub(r'\d+(\.\d+)?\s?★', '', str(name)).strip()).str.lower()

    return df

def process_price_column(df):
    """
    Process the 'price' column in the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with 'price' column processed.
    """
    # Exclude rows with 'Price on Request'
    df = df[df['price'] != 'Price on Request']

    def treat_price(x):
        if type(x) == float:
            return x
        else:
            # Convert prices to numeric values based on the unit (Lac or other)
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
    - pd.DataFrame: DataFrame with 'price_per_sqft' column cleaned.
    """
    # Extract the numeric part of 'price_per_sqft' and clean the formatting
    df['price_per_sqft'] = df['price_per_sqft'].str.split('/').str.get(0).str.replace('₹','').str.replace(',','').str.strip().astype('float')

    return df

def treat_bedroom_col(df):
    """
    Treat the 'bedRoom' column in the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with 'bedRoom' column treated.
    """
    # Remove rows with null values in 'bedRoom' column
    df = df[~df['bedRoom'].isnull()]

    # Extract the numeric part of 'bedRoom' and convert to integer
    df['bedRoom'] = df['bedRoom'].str.split(' ').str.get(0).astype('int')

    return df

def treat_bathroom_col(df):
    """
    Treat the 'bathroom' column in the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with 'bathroom' column treated.
    """
    # Extract the numeric part of 'bathroom' and convert to integer
    df['bathroom'] = df['bathroom'].str.split(' ').str.get(0).astype('int')

    return df

def treat_balcony_col(df):
    """
    Treat the 'balcony' column in the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with 'balcony' column treated.
    """
    # Extract the numeric part of 'balcony', replace 'No' with '0', and convert to string
    df['balcony'] = df['balcony'].str.split(' ').str.get(0).str.replace('No', '0')

    return df

def treat_additional_room_col(df):
    """
    Treat the 'additionalRoom' column in the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with 'additionalRoom' column treated.
    """
    # Fill NaN values with 'not available' and convert 'additionalRoom' to lowercase
    df['additionalRoom'].fillna('not available', inplace=True)
    df['additionalRoom'] = df['additionalRoom'].str.lower()

    return df

def treat_floorNum_col(df):
    """
    Treat the 'floorNum' column in the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with 'floorNum' column treated.
    """
    # Replace specific values and extract digits from 'floorNum' column
    df['floorNum'] = df['floorNum'].str.split(' ').str.get(0).replace({'Ground': '0', 'Basement': '-1', 'Lower': '0'}).str.extract(r'(\d+)')
    
    return df

def treat_facing_col(df):
    """
    Treat the 'facing' column in the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with 'facing' column treated.
    """
    # Fill missing values in 'facing' column with 'NA'
    df['facing'].fillna('NA', inplace=True)
    
    return df

def calculate_area_and_insert_col(df):
    """
    Calculate the 'area' column and insert it into the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with 'area' column calculated and inserted.
    """
    # Calculate the 'area' column and insert it into the DataFrame
    df.insert(loc=4, column='area', value=round((df['price'] * 10000000) / df['price_per_sqft']))
    
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

    # Load file-specific parameters for 'data-preprocessing-flats' from 'params.yaml'
    file_params = yaml.safe_load(open(params_file))["data-preprocessing-flats"]


    # input_file = sys.argv[1]
    # data_path = home_dir.as_posix() + input_file
    # output_path = home_dir.as_posix() + '/data/processed'


    # input_file = '/raw/flats.csv'
    # gdrive_folder_id = '1j5VgVtf-JdHt8aGf-MmCtsp-VLUJbtUL'  # Replace with the actual folder ID on Google Drive

    # drive = authenticate_drive()
    # data_path = f'gdrive://{gdrive_folder_id}/{input_file}'
    # output_path = f'gdrive://{gdrive_folder_id}/processed'

    

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
    data = drop_and_rename_cols(data)
    data = clean_society_col(data)
    data = process_price_column(data)
    data = clean_price_per_sqft_col(data)
    data = treat_bedroom_col(data)
    data = treat_bathroom_col(data)
    data = treat_balcony_col(data)
    data = treat_additional_room_col(data)
    data = treat_floorNum_col(data)
    data = treat_facing_col(data)
    data = calculate_area_and_insert_col(data)

    # Insert 'property_type' column with value 'flat'
    data.insert(loc=1,column='property_type',value='flat')

    # Save the processed data back to S3
    save_data(data=data,
              bucket=s3_bucket,
              key=output_s3_key,
              aws_access_key_id=aws_access_key_id,
              aws_secret_access_key=aws_secret_access_key,
              region_name=region_name)

if __name__ == "__main__":
    main()