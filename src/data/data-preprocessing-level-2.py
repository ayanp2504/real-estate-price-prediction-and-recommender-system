# data-preprocessing-level-2.py
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
#     # Save the split datasets to the specified output path
#     pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
#     data.to_csv(output_path + '/gurgaon_properties_cleaned_v1.csv', index=False)

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

def replace_places_with_their_sector(df):
    """
    Replace place names with their corresponding sectors in the given DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the 'sector' column.

    Returns:
    - pd.DataFrame: DataFrame with updated 'sector' column.
    """
    df['sector'] = df['sector'].str.lower()
    df['sector'] = df['sector'].str.replace('dharam colony','sector 12')
    df['sector'] = df['sector'].str.replace('krishna colony','sector 7')
    df['sector'] = df['sector'].str.replace('suncity','sector 54')
    df['sector'] = df['sector'].str.replace('prem nagar','sector 13')
    df['sector'] = df['sector'].str.replace('mg road','sector 28')
    df['sector'] = df['sector'].str.replace('gandhi nagar','sector 28')
    df['sector'] = df['sector'].str.replace('laxmi garden','sector 11')
    df['sector'] = df['sector'].str.replace('shakti nagar','sector 11')
    df['sector'] = df['sector'].str.replace('baldev nagar','sector 7')
    df['sector'] = df['sector'].str.replace('shivpuri','sector 7')
    df['sector'] = df['sector'].str.replace('garhi harsaru','sector 17')
    df['sector'] = df['sector'].str.replace('imt manesar','manesar')
    df['sector'] = df['sector'].str.replace('adarsh nagar','sector 12')
    df['sector'] = df['sector'].str.replace('shivaji nagar','sector 11')
    df['sector'] = df['sector'].str.replace('bhim nagar','sector 6')
    df['sector'] = df['sector'].str.replace('madanpuri','sector 7')
    df['sector'] = df['sector'].str.replace('saraswati vihar','sector 28')
    df['sector'] = df['sector'].str.replace('arjun nagar','sector 8')
    df['sector'] = df['sector'].str.replace('ravi nagar','sector 9')
    df['sector'] = df['sector'].str.replace('vishnu garden','sector 105')
    df['sector'] = df['sector'].str.replace('bhondsi','sector 11')
    df['sector'] = df['sector'].str.replace('surya vihar','sector 21')
    df['sector'] = df['sector'].str.replace('devilal colony','sector 9')
    df['sector'] = df['sector'].str.replace('valley view estate','gwal pahari')
    df['sector'] = df['sector'].str.replace('mehrauli  road','sector 14')
    df['sector'] = df['sector'].str.replace('jyoti park','sector 7')
    df['sector'] = df['sector'].str.replace('ansal plaza','sector 23')
    df['sector'] = df['sector'].str.replace('dayanand colony','sector 6')
    df['sector'] = df['sector'].str.replace('sushant lok phase 2','sector 55')
    df['sector'] = df['sector'].str.replace('chakkarpur','sector 28')
    df['sector'] = df['sector'].str.replace('greenwood city','sector 45')
    df['sector'] = df['sector'].str.replace('subhash nagar','sector 12')
    df['sector'] = df['sector'].str.replace('sohna road road','sohna road')
    df['sector'] = df['sector'].str.replace('malibu town','sector 47')
    df['sector'] = df['sector'].str.replace('surat nagar 1','sector 104')
    df['sector'] = df['sector'].str.replace('new colony','sector 7')
    df['sector'] = df['sector'].str.replace('mianwali colony','sector 12')
    df['sector'] = df['sector'].str.replace('jacobpura','sector 12')
    df['sector'] = df['sector'].str.replace('rajiv nagar','sector 13')
    df['sector'] = df['sector'].str.replace('ashok vihar','sector 3')
    df['sector'] = df['sector'].str.replace('dlf phase 1','sector 26')
    df['sector'] = df['sector'].str.replace('nirvana country','sector 50')
    df['sector'] = df['sector'].str.replace('palam vihar','sector 2')
    df['sector'] = df['sector'].str.replace('dlf phase 2','sector 25')
    df['sector'] = df['sector'].str.replace('sushant lok phase 1','sector 43')
    df['sector'] = df['sector'].str.replace('laxman vihar','sector 4')
    df['sector'] = df['sector'].str.replace('dlf phase 4','sector 28')
    df['sector'] = df['sector'].str.replace('dlf phase 3','sector 24')
    df['sector'] = df['sector'].str.replace('sushant lok phase 3','sector 57')
    df['sector'] = df['sector'].str.replace('dlf phase 5','sector 43')
    df['sector'] = df['sector'].str.replace('rajendra park','sector 105')
    df['sector'] = df['sector'].str.replace('uppals southend','sector 49')
    df['sector'] = df['sector'].str.replace('sohna','sohna road')
    df['sector'] = df['sector'].str.replace('ashok vihar phase 3 extension','sector 5')
    df['sector'] = df['sector'].str.replace('south city 1','sector 41')
    df['sector'] = df['sector'].str.replace('ashok vihar phase 2','sector 5')
    df['sector'] = df['sector'].str.replace('sector 95a','sector 95')
    df['sector'] = df['sector'].str.replace('sector 23a','sector 23')
    df['sector'] = df['sector'].str.replace('sector 12a','sector 12')
    df['sector'] = df['sector'].str.replace('sector 3a','sector 3')
    df['sector'] = df['sector'].str.replace('sector 110 a','sector 110')
    df['sector'] = df['sector'].str.replace('patel nagar','sector 15')
    df['sector'] = df['sector'].str.replace('a block sector 43','sector 43')
    df['sector'] = df['sector'].str.replace('maruti kunj','sector 12')
    df['sector'] = df['sector'].str.replace('b block sector 43','sector 43')
    df['sector'] = df['sector'].str.replace('sector-33 sohna road','sector 33')
    df['sector'] = df['sector'].str.replace('sector 1 manesar','manesar')
    df['sector'] = df['sector'].str.replace('sector 4 phase 2','sector 4')
    df['sector'] = df['sector'].str.replace('sector 1a manesar','manesar')
    df['sector'] = df['sector'].str.replace('c block sector 43','sector 43')
    df['sector'] = df['sector'].str.replace('sector 89 a','sector 89')
    df['sector'] = df['sector'].str.replace('sector 2 extension','sector 2')
    df['sector'] = df['sector'].str.replace('sector 36 sohna road','sector 36')
    return df

def filter_sector_counts(df, threshold=3):
    """
    Filter DataFrame to include only sectors with counts equal to or above a given threshold.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the 'sector' column.
    - threshold (int, optional): Minimum count threshold. Defaults to 3.

    Returns:
    - pd.DataFrame: Filtered DataFrame.
    """
    sector_counts = df['sector'].value_counts()
    selected_sectors = sector_counts[sector_counts >= threshold].index
    df = df[df['sector'].isin(selected_sectors)]
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
    file_params = yaml.safe_load(open(params_file))["data-preprocessing-level-2"]

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

    # curr_dir = pathlib.Path(__file__)
    # home_dir = curr_dir.parent.parent.parent

    # input_file = sys.argv[1]
    # data_path = home_dir.as_posix() + input_file
    # output_path = home_dir.as_posix() + '/data/processed'
    # data = load_data(data_path)

    # Insert sector column
    data.insert(loc=3,column='sector',value=data['property_name'].str.split('in').str.get(1).str.replace('Gurgaon','').str.strip())

    data = replace_places_with_their_sector(data)

    # Manually changing some sectors
    data.loc[955,'sector'] = 'sector 37'
    data.loc[2800,'sector'] = 'sector 92'
    data.loc[2838,'sector'] = 'sector 90'
    data.loc[2857,'sector'] = 'sector 76'
    data.loc[[311,1072,1486,3040,3875],'sector'] = 'sector 110'

    
    # features to drop -> property_name, address, description, rating
    data.drop(columns=['property_name', 'address', 'description', 'rating'],inplace=True)

    data = filter_sector_counts(data,  3)


    # Save the processed data back to S3
    save_data(data=data,
              bucket=s3_bucket,
              key=output_s3_key,
              aws_access_key_id=aws_access_key_id,
              aws_secret_access_key=aws_secret_access_key,
              region_name=region_name)


if __name__ == "__main__":
    main()