import pandas as pd
import sys
import ast
import pickle
import pathlib
import boto3
import yaml
from io import StringIO, BytesIO
from wordcloud import WordCloud

# def load_data(data_path):
#     # Load your dataset from a given path
#     df = pd.read_csv(data_path)
#     return df

# def save_data(data, output_path, file_name):
#     # Save the split datasets to the specified output path
#     pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
#     data.to_csv(output_path + file_name, index=False)
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



def save_pickled_model(model, bucket, key, aws_access_key_id=None, aws_secret_access_key=None, region_name=None):
    """
    Save a pickled model to an S3 bucket.

    Parameters:
    - model: The model or data to be pickled and saved.
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

    # Pickle the model into bytes
    pickled_model = pickle.dumps(model)

    # Upload pickled model to S3 object
    s3.put_object(Bucket=bucket, Key=key, Body=BytesIO(pickled_model).read())


def sep_lat_long(latlong):
    '''This function coordinates into its latitude and longitude'''
    latlong['latitude'] = latlong['coordinates'].str.split(',').str.get(0).str.split('°').str.get(0).astype('float')
    latlong['longitude'] = latlong['coordinates'].str.split(',').str.get(1).str.split('°').str.get(0).astype('float')
    return latlong

def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    # Define the path to the 'params.yaml' file within the home directory
    params_file = home_dir.as_posix() + '/params.yaml'

    # Load S3 parameters from 'params.yaml'
    s3_params = yaml.safe_load(open(params_file))["s3"]

    # Load file-specific parameters for 'data-preprocessing-flats' from 'params.yaml'
    file_params = yaml.safe_load(open(params_file))["data_visualization"]

     # Extract S3 parameters from the loaded 's3_params'
    s3_bucket = s3_params['bucket']
    s3_key1 = file_params['input_data1']
    s3_key2 = file_params['input_data2']
    s3_key3 = file_params['input_data3']
    output_s3_key1 = file_params['output_data']
    output_s3_key2 = file_params['output_model']
    aws_access_key_id = s3_params['aws_access_key_id']
    aws_secret_access_key = s3_params['aws_secret_access_key']
    region_name = s3_params['region_name']

    # input_file1 = sys.argv[1]
    # data_path1 = home_dir.as_posix() + input_file1
    # latlong = load_data(data_path1)
    # latlong = sep_lat_long(latlong)

    # Load data from S3 using specified parameters
    latlong = load_data(bucket=s3_bucket,
                     key=s3_key1,
                     aws_access_key_id=aws_access_key_id,
                     aws_secret_access_key=aws_secret_access_key,
                     region_name=region_name)
    
    latlong = sep_lat_long(latlong)

    # Merge latlong data with main dataset
    # input_file2 = sys.argv[2]
    # data_path2 = home_dir.as_posix() + input_file2

    df = load_data(bucket=s3_bucket,
                     key=s3_key2,
                     aws_access_key_id=aws_access_key_id,
                     aws_secret_access_key=aws_secret_access_key,
                     region_name=region_name)
    # df = load_data(data_path2)
    new_df = df.merge(latlong, on='sector')
    # output_path = home_dir.as_posix() + '/data/processed'
    # save_data(new_df, output_path, '/data_viz1.csv')

    # Save the processed data back to S3
    save_data(data=new_df,
              bucket=s3_bucket,
              key=output_s3_key1,
              aws_access_key_id=aws_access_key_id,
              aws_secret_access_key=aws_secret_access_key,
              region_name=region_name)

    # input_file3= sys.argv[3]
    # data_path3 = home_dir.as_posix() + input_file3
    # df1 = load_data(data_path3)

    df1 = load_data(bucket=s3_bucket,
                     key=s3_key3,
                     aws_access_key_id=aws_access_key_id,
                     aws_secret_access_key=aws_secret_access_key,
                     region_name=region_name)
    
    wordcloud_df = df1.merge(df, left_index=True, right_index=True)[['features','sector']]
    main = []
    for item in wordcloud_df['features'].dropna().apply(ast.literal_eval):
        main.extend(item)
    feature_text = ' '.join(main)

    # file_path = home_dir.as_posix() + '/models/feature_text.pkl'

    
    # # Dump the variable to the specified path
    # with open(file_path, 'wb') as file:
    #     pickle.dump(feature_text, file)

    save_pickled_model(model=feature_text,
            bucket=s3_bucket,
            key=output_s3_key2,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name)


if __name__ == "__main__":
    main()