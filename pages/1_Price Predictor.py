import streamlit as st
import pickle
import pandas as pd
import numpy as np
import pathlib

# Set Streamlit page configuration
st.set_page_config(page_title="Price Predictor")

curr_dir = pathlib.Path(__file__)
home_dir = curr_dir.parent.parent.as_posix()

# Load the DataFrame and pipeline model from pickle files
df = pd.read_csv(home_dir + '/data/processed/gurgaon_properties_post_feature_selection.csv')

with open(home_dir + '/models/trained_model.pkl', 'rb') as file:
    pipeline = pickle.load(file)

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
