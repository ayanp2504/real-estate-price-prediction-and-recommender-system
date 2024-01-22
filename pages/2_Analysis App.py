import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import pathlib
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import boto3
import yaml
import botocore
from io import BytesIO

# Set Streamlit page configuration
st.set_page_config(page_title="Analytics App")


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

    try:
        # Get S3 object
        obj = s3.get_object(Bucket=bucket, Key=key)

        # Read CSV data from S3 object's body
        df = pd.read_csv(obj['Body'])
        return df

    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            print(f"Error: The specified key '{key}' does not exist in the S3 bucket.")
            return None
        else:
            print(f"Error: {e}")
            return None

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
s3_key = file_params['analytics_data']
s3_analytics_model= file_params['analytics_model_data']
aws_access_key_id = s3_params['aws_access_key_id']
aws_secret_access_key = s3_params['aws_secret_access_key']
region_name = s3_params['region_name']


# Streamlit app title
st.title('Analytics')

# Read the CSV file and load the feature_text from pickle
new_df = load_data(bucket=s3_bucket,
                    key=s3_key,
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=region_name)
feature_text = load_pickled_model(bucket=s3_bucket,
        key=s3_analytics_model,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name)

st.set_option('deprecation.showPyplotGlobalUse', False)
# Group data by 'sector' for Geomap
group_df = new_df.groupby('sector').mean()[['price', 'price_per_sqft', 'built_up_area', 'latitude', 'longitude']]

# Geomap section
st.header('Sector Price per Sqft Geomap')
fig = px.scatter_mapbox(group_df, lat="latitude", lon="longitude", color="price_per_sqft", size='built_up_area',
                  color_continuous_scale=px.colors.cyclical.IceFire, zoom=10,
                  mapbox_style="open-street-map", width=1200, height=700, hover_name=group_df.index)

st.plotly_chart(fig, use_container_width=True)

# Wordcloud section
st.header('Features Wordcloud')
wordcloud = WordCloud(width=800, height=800,
                      background_color='black',
                      stopwords=set(['s']),  # Any stopwords you'd like to exclude
                      min_font_size=10).generate(feature_text)

plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
st.pyplot()

# Area Vs Price section
st.header('Area Vs Price')
property_type = st.selectbox('Select Property Type', ['flat', 'house'])

if property_type == 'house':
    fig1 = px.scatter(new_df[new_df['property_type'] == 'house'], x="built_up_area", y="price", color="bedRoom", title="Area Vs Price")
    st.plotly_chart(fig1, use_container_width=True)
else:
    fig1 = px.scatter(new_df[new_df['property_type'] == 'flat'], x="built_up_area", y="price", color="bedRoom",
                      title="Area Vs Price")
    st.plotly_chart(fig1, use_container_width=True)

# BHK Pie Chart section
st.header('BHK Pie Chart')
sector_options = new_df['sector'].unique().tolist()
sector_options.insert(0, 'overall')
selected_sector = st.selectbox('Select Sector', sector_options)

if selected_sector == 'overall':
    fig2 = px.pie(new_df, names='bedRoom')
    st.plotly_chart(fig2, use_container_width=True)
else:
    fig2 = px.pie(new_df[new_df['sector'] == selected_sector], names='bedRoom')
    st.plotly_chart(fig2, use_container_width=True)

# Side by Side BHK price comparison section
st.header('Side by Side BHK price comparison')
fig3 = px.box(new_df[new_df['bedRoom'] <= 4], x='bedRoom', y='price', title='BHK Price Range')
st.plotly_chart(fig3, use_container_width=True)

# Side by Side Distplot for property type section
st.header('Side by Side Distplot for property type')
fig3 = plt.figure(figsize=(10, 4))
sns.distplot(new_df[new_df['property_type'] == 'house']['price'], label='house')
sns.distplot(new_df[new_df['property_type'] == 'flat']['price'], label='flat')
plt.legend()
st.pyplot(fig3)
