import streamlit as st
import pickle
import pathlib
import pandas as pd
import numpy as np

# Set Streamlit page configuration
st.set_page_config(page_title="Recommend Appartments")

curr_dir = pathlib.Path(__file__)
home_dir = curr_dir.parent.parent.as_posix()

# Load location_df and cosine similarity matrices from pickle files
location_df = pickle.load(open(home_dir + '/models/location_df.pkl', 'rb'))
cosine_sim1 = pickle.load(open(home_dir + '/models/cosine_sim1.pkl', 'rb'))
cosine_sim2 = pickle.load(open(home_dir + '/models/cosine_sim2.pkl', 'rb'))
cosine_sim3 = pickle.load(open(home_dir + '/models/cosine_sim3.pkl', 'rb'))

# Define a function to recommend properties with scores
def recommend_properties_with_scores(property_name, top_n=5):
    cosine_sim_matrix = 0.5 * cosine_sim1 + 0.8 * cosine_sim2 + 1 * cosine_sim3
    # cosine_sim_matrix = cosine_sim3

    # Get the similarity scores for the property using its name as the index
    sim_scores = list(enumerate(cosine_sim_matrix[location_df.index.get_loc(property_name)]))

    # Sort properties based on the similarity scores
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices and scores of the top_n most similar properties
    top_indices = [i[0] for i in sorted_scores[1:top_n + 1]]
    top_scores = [i[1] for i in sorted_scores[1:top_n + 1]]

    # Retrieve the names of the top properties using the indices
    top_properties = location_df.index[top_indices].tolist()

    # Create a dataframe with the results
    recommendations_df = pd.DataFrame({
        'PropertyName': top_properties,
        'SimilarityScore': top_scores
    })

    return recommendations_df

# Test the recommender function using a property name
recommend_properties_with_scores('DLF The Camellias')

# Streamlit app title
st.title('Select Location and Radius')

# Location selection and radius input
selected_location = st.selectbox('Location', sorted(location_df.columns.to_list()))
radius = st.number_input('Radius in Kms')

# Search button to find properties within the selected radius
if st.button('Search'):
    result_ser = location_df[location_df[selected_location] < radius * 1000][selected_location].sort_values()

    # Display the search results
    for key, value in result_ser.items():
        st.text(str(key) + " " + str(round(value / 1000)) + ' kms')

# Streamlit app title for property recommendation
st.title('Recommend Appartments')

# Property selection dropdown
selected_appartment = st.selectbox('Select an appartment', sorted(location_df.index.to_list()))

# Recommend button to find similar properties
if st.button('Recommend'):
    recommendation_df = recommend_properties_with_scores(selected_appartment)

    # Display the recommendation results
    st.dataframe(recommendation_df)
