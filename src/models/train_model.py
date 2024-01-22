import sys
import pathlib
import pickle
import pandas as pd
import boto3
import yaml
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

# def load_data(data_path):
#     # Load your dataset from a given path
#     df = pd.read_csv(data_path)
#     return df

# def save_model(model, output_path):
#     # Save the split datasets to the specified output path
#     pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
#     model_path = output_path + '/trained_model.pkl'
#     with open(model_path, 'wb') as model_file:
#         pickle.dump(model, model_file)
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

def save_pickle_model(model, bucket, key, aws_access_key_id=None, aws_secret_access_key=None, region_name=None):
    """
    Save a pickled model to an S3 bucket.

    Parameters:
    - model: The model object to be saved using pickle.
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
    model_bytes = pickle.dumps(model)
    
    # Upload pickled model to S3 object
    s3.put_object(Bucket=bucket, Key=key, Body=BytesIO(model_bytes).read())

def train_model(df):
    """
    Train a RandomForestRegressor model on the provided DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the training data.

    Returns:
    - pipeline: Trained RandomForestRegressor model pipeline.
    """
    # Separate features (X) and target variable (y)
    X = df.drop(columns=['price'])
    y = df['price']

    # Columns to encode using different encoders
    columns_to_encode = ['property_type', 'sector', 'balcony', 'agePossession', 'furnishing_type', 'luxury_category', 'floor_category']

    # Preprocess the data using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']),
            ('cat', OrdinalEncoder(), columns_to_encode),
            ('cat1', OneHotEncoder(drop='first', sparse=False), ['sector', 'agePossession'])
        ],
        remainder='passthrough'
    )

    # Hyperparameters for RandomForestRegressor
    random_forest_hyperparameters = {
        'n_estimators': 300,
        'max_depth': 30,
        'max_features': 'sqrt',
        'max_samples': 1.0
    }

    # Create the pipeline with RandomForestRegressor and hyperparameters
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=random_forest_hyperparameters['n_estimators'],
                                            max_depth=random_forest_hyperparameters['max_depth'],
                                            max_features=random_forest_hyperparameters['max_features'],
                                            max_samples=random_forest_hyperparameters['max_samples']))
    ])

    # Fit the data
    pipeline.fit(X, y)

    return pipeline

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
    file_params = yaml.safe_load(open(params_file))["train_model"]
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
    
    model = train_model(df)
    # Save the processed data back to S3
    save_pickle_model(model=model,
              bucket=s3_bucket,
              key=output_s3_key,
              aws_access_key_id=aws_access_key_id,
              aws_secret_access_key=aws_secret_access_key,
              region_name=region_name)

    # output_path = home_dir.as_posix() + '/models'
    # save_model(model, output_path)


if __name__ == "__main__":
    main()