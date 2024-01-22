# missing_value_imputation.py 
import pathlib
import sys
import numpy as np
import pandas as pd
import boto3
import yaml
from io import StringIO

# def load_data(data_path):
#     # Load your dataset from a given path
#     df = pd.read_csv(data_path)
#     return df

# def save_data(data, output_path):
#     # Save the split datasets to the specified output path
#     pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
#     data.to_csv(output_path + '/gurgaon_properties_missing_value_imputation.csv', index=False)
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

def mode_based_imputation(row, df):
    """
    Impute missing values in the 'agePossession' column based on mode values from similar rows.

    Parameters:
    - row (pd.Series): Row containing the 'agePossession' column to be imputed.
    - df (pd.DataFrame): DataFrame containing the data.

    Returns:
    - str or np.nan: Imputed value for the 'agePossession' column.
    """
    if row['agePossession'] == 'Undefined':
        mode_value = df[(df['sector'] == row['sector']) & (df['property_type'] == row['property_type'])]['agePossession'].mode()
        # If mode_value is empty (no mode found), return NaN, otherwise return the mode
        if not mode_value.empty:
            return mode_value.iloc[0] 
        else:
            return np.nan
    else:
        return row['agePossession']

    
def mode_based_imputation2(row, df):
    """
    Impute missing values in the 'agePossession' column based on mode values from similar rows.

    Parameters:
    - row (pd.Series): Row containing the 'agePossession' column to be imputed.
    - df (pd.DataFrame): DataFrame containing the data.

    Returns:
    - str or np.nan: Imputed value for the 'agePossession' column.
    """
    if row['agePossession'] == 'Undefined':
        mode_value = df[(df['sector'] == row['sector'])]['agePossession'].mode()
        # If mode_value is empty (no mode found), return NaN, otherwise return the mode
        if not mode_value.empty:
            return mode_value.iloc[0] 
        else:
            return np.nan
    else:
        return row['agePossession']

def mode_based_imputation3(row, df):
    """
    Impute missing values in the 'agePossession' column based on mode values from similar rows.

    Parameters:
    - row (pd.Series): Row containing the 'agePossession' column to be imputed.
    - df (pd.DataFrame): DataFrame containing the data.

    Returns:
    - str or np.nan: Imputed value for the 'agePossession' column.
    """
    if row['agePossession'] == 'Undefined':
        mode_value = df[(df['property_type'] == row['property_type'])]['agePossession'].mode()
        # If mode_value is empty (no mode found), return NaN, otherwise return the mode
        if not mode_value.empty:
            return mode_value.iloc[0] 
        else:
            return np.nan
    else:
        return row['agePossession']


def main():

    # curr_dir = pathlib.Path(__file__)
    # home_dir = curr_dir.parent.parent.parent

    # input_file = sys.argv[1]
    # data_path = home_dir.as_posix() + input_file
    # df = load_data(data_path)

    # Get the current directory path
    curr_dir = pathlib.Path(__file__)

    # Navigate up three levels to reach the parent directory
    home_dir = curr_dir.parent.parent.parent

    # Define the path to the 'params.yaml' file within the home directory
    params_file = home_dir.as_posix() + '/params.yaml'

    # Load S3 parameters from 'params.yaml'
    s3_params = yaml.safe_load(open(params_file))["s3"]

    # Load file-specific parameters for 'data-preprocessing-flats' from 'params.yaml'
    file_params = yaml.safe_load(open(params_file))["missing_value_imputation"]

    # Extract S3 parameters from the loaded 's3_params'
    s3_bucket = s3_params['bucket']
    s3_key = file_params['input_data']
    output_s3_key = file_params['output_data']
    aws_access_key_id = s3_params['aws_access_key_id']
    aws_secret_access_key = s3_params['aws_secret_access_key']
    region_name = s3_params['region_name']
    
    # Load data from S3 using specified parameters
    df = load_data(bucket=s3_bucket,
                     key=s3_key,
                     aws_access_key_id=aws_access_key_id,
                     aws_secret_access_key=aws_secret_access_key,
                     region_name=region_name)

    # Filter rows where all three area columns are present (super_built_up_area, built_up_area, carpet_area)
    all_present_df = df[~((df['super_built_up_area'].isnull()) | (df['built_up_area'].isnull()) | (df['carpet_area'].isnull()))]

    # Calculate the median ratio of super_built_up_area to built_up_area and carpet_area to built_up_area
    super_to_built_up_ratio = (all_present_df['super_built_up_area'] / all_present_df['built_up_area']).median()
    carpet_to_built_up_ratio = (all_present_df['carpet_area'] / all_present_df['built_up_area']).median()

    # Filter rows where super_built_up_area is present, built_up_area is null, and carpet_area is present
    sbc_df = df[~(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & ~(df['carpet_area'].isnull())]

    # Fill missing values in built_up_area using a weighted average of super_built_up_area and carpet_area
    sbc_df['built_up_area'].fillna(round(((sbc_df['super_built_up_area'] / 1.105) + (sbc_df['carpet_area'] / 0.9)) / 2), inplace=True)

    # Update the original DataFrame with the filled values
    df.update(sbc_df)


    # Filter rows where super_built_up_area is present, built_up_area is null, and carpet_area is null
    sb_df = df[~(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & (df['carpet_area'].isnull())]

    # Fill missing values in built_up_area using a weighted average of super_built_up_area
    sb_df['built_up_area'].fillna(round(sb_df['super_built_up_area'] / 1.105), inplace=True)

    # Update the original DataFrame with the filled values
    df.update(sb_df)

    # Filter rows where super_built_up_area is null, built_up_area is null, and carpet_area is present
    c_df = df[(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & ~(df['carpet_area'].isnull())]

    # Fill missing values in built_up_area using a weighted average of carpet_area
    c_df['built_up_area'].fillna(round(c_df['carpet_area'] / 0.9), inplace=True)

    # Update the original DataFrame with the filled values
    df.update(c_df)


    # Filter rows with anomalies (built_up_area < 2000 and price > 2.5)
    anomaly_df = df[(df['built_up_area'] < 2000) & (df['price'] > 2.5)][['price', 'area', 'built_up_area']]

    # Replace the 'built_up_area' values in the anomaly_df with 'area'
    anomaly_df['built_up_area'] = anomaly_df['area']

    # Update the original DataFrame with the corrected values
    df.update(anomaly_df)

    # Drop unnecessary columns from the original DataFrame
    df.drop(columns=['area', 'areaWithType', 'super_built_up_area', 'carpet_area', 'area_room_ratio'], inplace=True)


    ################ floorNum ###########################

    # Calculate the median floorNum for rows with property_type 'house'
    median_floor_num_house = df[df['property_type'] == 'house']['floorNum'].median()

    # Fill missing values in 'floorNum' with the calculated median value (2.0)
    df['floorNum'].fillna(median_floor_num_house, inplace=True)


    ################# facing #######################

    # Drop the 'facing' column from the DataFrame
    df.drop(columns=['facing'], inplace=True)

    # Drop the row with index 2536
    df.drop(index=[2536], inplace=True)


    ################# agePossession #################
    # Filter rows where 'agePossession' is 'Undefined'
    # undefined_age_df = df[df['agePossession'] == 'Undefined']

    # Impute missing values in 'agePossession' using mode-based imputation with different criteria
    df['agePossession'] = df.apply(mode_based_imputation, args=(df,), axis=1)
    df['agePossession'] = df.apply(mode_based_imputation2,args=(df,), axis=1)
    df['agePossession'] = df.apply(mode_based_imputation3,args=(df,), axis=1)

    # Save the processed data back to S3
    save_data(data=df,
              bucket=s3_bucket,
              key=output_s3_key,
              aws_access_key_id=aws_access_key_id,
              aws_secret_access_key=aws_secret_access_key,
              region_name=region_name)

if __name__ == "__main__":
    main()