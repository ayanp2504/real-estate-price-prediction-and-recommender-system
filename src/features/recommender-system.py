import numpy as np
import pandas as pd
import pathlib
import sys
import re
import json
import ast
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(data_path):
    # Load your dataset from a given path
    df = pd.read_csv(data_path)
    return df

def save_data(data, output_path):
    # Save the split datasets to the specified output path
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path + '/gurgaon_properties_cleaned_v2.csv', index=False)

def extract_list(s):
    """
    Extracts a list of strings enclosed in single quotes from the given input string.

    Parameters:
    - s (str): Input string containing items enclosed in single quotes.

    Returns:
    - List[str]: A list of strings extracted from the input string.
    """
    return re.findall(r"'(.*?)'", s)

def calculate_tfidf_matrix(data_frame, column_name, stop_words='english', ngram_range=(1, 2)):
    """
    Calculate the TF-IDF matrix for a specified column in a DataFrame.

    Parameters:
    - data_frame (pd.DataFrame): The DataFrame containing the text data.
    - column_name (str): The name of the column containing text data.
    - stop_words (str or None, default='english'): The stopwords parameter for TfidfVectorizer.
    - ngram_range (tuple, default=(1, 2)): The ngram_range parameter for TfidfVectorizer.

    Returns:
    - np.ndarray: The TF-IDF matrix for the specified column.
    """

    # Create a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=ngram_range)

    # Transform the specified column into a TF-IDF matrix
    tfidf_matrix = tfidf_vectorizer.fit_transform(data_frame[column_name])

    return tfidf_matrix.toarray()

def calculate_cosine_similarity(tfidf_matrix):
    """
    Calculate cosine similarity based on the TF-IDF matrix.

    Parameters:
    - tfidf_matrix (np.ndarray): The TF-IDF matrix.

    Returns:
    - np.ndarray: The cosine similarity matrix.
    """

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return cosine_sim

def recommend_properties(df, property_name, cosine_sim):
    """
    Recommends similar properties based on cosine similarity.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the text data.
    - property_name (str): The name of the property for which recommendations are sought.
    - cosine_sim (np.ndarray): The cosine similarity matrix.

    Returns:
    - pd.DataFrame: A DataFrame containing recommended properties and their similarity scores.
    """

    # Get the index of the property that matches the name
    idx = df.index[df['PropertyName'] == property_name].tolist()[0]

    # Get the pairwise similarity scores with that property
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the properties based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 5 most similar properties (excluding the property itself)
    sim_scores = sim_scores[1:6]

    # Get the property indices
    property_indices = [i[0] for i in sim_scores]
    
    # Create a DataFrame with recommended properties and their similarity scores
    recommendations_df = pd.DataFrame({
        'PropertyName': df['PropertyName'].iloc[property_indices],
        'SimilarityScore': [i[1] for i in sim_scores]
    })

    # Return the top 5 most similar properties
    return recommendations_df

def refined_parse_modified_v2(detail_str):
    """
    Parse and extract features from the PriceDetails column in a refined manner.

    Parameters:
    - detail_str (str): The input string containing price and area details in JSON format.

    Returns:
    - dict: A dictionary containing the extracted features.
    """

    try:
        # Convert the JSON-like string to a dictionary
        details = json.loads(detail_str.replace("'", "\""))
    except:
        return {}

    # Initialize an empty dictionary to store extracted features
    extracted = {}

    # Iterate through each BHK configuration in the details
    for bhk, detail in details.items():
        # Extract building type
        extracted[f'building type_{bhk}'] = detail.get('building_type')

        # Parsing area details
        area = detail.get('area', '')
        area_parts = area.split('-')
        if len(area_parts) == 1:
            try:
                value = float(area_parts[0].replace(',', '').replace(' sq.ft.', '').strip())
                extracted[f'area low {bhk}'] = value
                extracted[f'area high {bhk}'] = value
            except:
                extracted[f'area low {bhk}'] = None
                extracted[f'area high {bhk}'] = None
        elif len(area_parts) == 2:
            try:
                extracted[f'area low {bhk}'] = float(area_parts[0].replace(',', '').replace(' sq.ft.', '').strip())
                extracted[f'area high {bhk}'] = float(area_parts[1].replace(',', '').replace(' sq.ft.', '').strip())
            except:
                extracted[f'area low {bhk}'] = None
                extracted[f'area high {bhk}'] = None

        # Parsing price details
        price_range = detail.get('price-range', '')
        price_parts = price_range.split('-')
        if len(price_parts) == 2:
            try:
                extracted[f'price low {bhk}'] = float(price_parts[0].replace('₹', '').replace(' Cr', '').replace(' L', '').strip())
                extracted[f'price high {bhk}'] = float(price_parts[1].replace('₹', '').replace(' Cr', '').replace(' L', '').strip())
                if 'L' in price_parts[0]:
                    extracted[f'price low {bhk}'] /= 100
                if 'L' in price_parts[1]:
                    extracted[f'price high {bhk}'] /= 100
            except:
                extracted[f'price low {bhk}'] = None
                extracted[f'price high {bhk}'] = None

    return extracted

def one_hot_encode_dataframe(df):
    """
    Perform one-hot encoding on categorical columns of a DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame after one-hot encoding.
    """
    # Identify categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    # Perform one-hot encoding
    ohe_df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Fill NaN values with 0
    ohe_df.fillna(0, inplace=True)

    return ohe_df

def recommend_properties_with_scores(property_name, df, cosine_sim_matrix, top_n=247):
    """
    Recommends properties based on cosine similarity scores for a given property.

    Parameters:
    - property_name (str): The name of the target property.
    - df (pd.DataFrame): The DataFrame containing property information.
    - cosine_sim_matrix (np.ndarray): The cosine similarity matrix.
    - top_n (int, default=247): The number of top similar properties to recommend.

    Returns:
    - pd.DataFrame: A DataFrame with recommended properties and their similarity scores.
    """

    # Ensure property_name is present in the DataFrame
    if property_name not in df.index:
        raise ValueError(f"The property '{property_name}' is not present in the DataFrame.")

    # Get the similarity scores for the property using its name as the index
    sim_scores = list(enumerate(cosine_sim_matrix[df.index.get_loc(property_name)]))
    
    # Sort properties based on the similarity scores
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the indices and scores of the top_n most similar properties
    top_indices = [i[0] for i in sorted_scores[1:top_n+1]]
    top_scores = [i[1] for i in sorted_scores[1:top_n+1]]
    
    # Retrieve the names of the top properties using the indices
    top_properties = df.index[top_indices].tolist()
    
    # Create a DataFrame with the results
    recommendations_df = pd.DataFrame({
        'PropertyName': top_properties,
        'SimilarityScore': top_scores
    })
    
    return recommendations_df

def distance_to_meters(distance_str):
    """
    Convert a distance string to meters.

    Parameters:
    - distance_str (str): The input distance string, e.g., '5 Km' or '300 Meter'.

    Returns:
    - float or None: The distance in meters, or None if the conversion fails.
    """

    try:
        # Check if the input string contains 'Km' or 'KM'
        if 'Km' in distance_str or 'KM' in distance_str:
            return float(distance_str.split()[0]) * 1000
        # Check if the input string contains 'Meter' or 'meter'
        elif 'Meter' in distance_str or 'meter' in distance_str:
            return float(distance_str.split()[0])
        else:
            return None
    except:
        return None

def recommend_properties_with_weighted_scores(df, property_name, cosine_matrices, weights, top_n=247):
    """
    Recommends properties based on weighted cosine similarity scores for a given property.

    Parameters:
    - DataFrame
    - property_name (str): The name of the target property.
    - cosine_matrices (list): List of cosine similarity matrices.
    - weights (list): List of weights corresponding to each cosine similarity matrix.
    - top_n (int, default=247): The number of top similar properties to recommend.

    Returns:
    - pd.DataFrame: A DataFrame with recommended properties and their similarity scores.
    """

    # Combine weighted cosine similarity matrices
    weighted_cosine_sim_matrix = sum(weight * cosine_matrix for weight, cosine_matrix in zip(weights, cosine_matrices))

    # Get the similarity scores for the property using its name as the index
    sim_scores = list(enumerate(weighted_cosine_sim_matrix[df.index.get_loc(property_name)]))
    
    # Sort properties based on the similarity scores
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the indices and scores of the top_n most similar properties
    top_indices = [i[0] for i in sorted_scores[1:top_n+1]]
    top_scores = [i[1] for i in sorted_scores[1:top_n+1]]
    
    # Retrieve the names of the top properties using the indices
    top_properties = df.index[top_indices].tolist()
    
    # Create a DataFrame with the results
    recommendations_df = pd.DataFrame({
        'PropertyName': top_properties,
        'SimilarityScore': top_scores
    })
    
    return recommendations_df


def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    input_file = '\\data\\raw\\appartments.csv'
    data_path = home_dir.as_posix() + input_file
    df = load_data(data_path).drop(22)

    # Apply the extract_list function to the 'TopFacilities' column
    df['TopFacilities'] = df['TopFacilities'].apply(extract_list)

    # Apply ' '.join to concatenate lists of strings in the 'TopFacilities' column
    df['FacilitiesStr'] = df['TopFacilities'].apply(' '.join)

    # Calculate the TF-IDF matrix
    tfidf_matrix = calculate_tfidf_matrix(df, 'FacilitiesStr')

    # Calculate cosine similarity
    cosine_sim1 = calculate_cosine_similarity(tfidf_matrix)


    # Apply the refined parsing and generate the new DataFrame structure
    data_refined = []

    # Iterate through each row in the original DataFrame
    for _, row in df.iterrows():
        # Parse and extract features using the refined parsing function
        features = refined_parse_modified_v2(row['PriceDetails'])
        
        # Construct a new row for the transformed DataFrame
        new_row = {'PropertyName': row['PropertyName']}
        
        # Populate the new row with extracted features
        for config in ['1 BHK', '2 BHK', '3 BHK', '4 BHK', '5 BHK', '6 BHK', '1 RK', 'Land']:
            new_row[f'building type_{config}'] = features.get(f'building type_{config}')
            new_row[f'area low {config}'] = features.get(f'area low {config}')
            new_row[f'area high {config}'] = features.get(f'area high {config}')
            new_row[f'price low {config}'] = features.get(f'price low {config}')
            new_row[f'price high {config}'] = features.get(f'price high {config}')
        
        # Append the new row to the list of refined data
        data_refined.append(new_row)

    # Create the final refined DataFrame and set 'PropertyName' as the index
    df_final_refined_v2 = pd.DataFrame(data_refined).set_index('PropertyName')

    # Replace empty values in the 'building type_Land' column with 'Land'
    df_final_refined_v2['building type_Land'] = df_final_refined_v2['building type_Land'].replace({'':'Land'})

    # Assuming you have a DataFrame 'df_final_refined_v2'
    ohe_df = one_hot_encode_dataframe(df_final_refined_v2)

    # Initialize the scaler
    scaler = StandardScaler()

    # Apply the scaler to the entire dataframe
    ohe_df_normalized = pd.DataFrame(scaler.fit_transform(ohe_df), columns=ohe_df.columns, index=ohe_df.index)

    # Compute the cosine similarity matrix
    cosine_sim2 = cosine_similarity(ohe_df_normalized)

    # Extract distances for each location
    location_matrix = {}
    for index, row in df.iterrows():
        distances = {}
        for location, distance in ast.literal_eval(row['LocationAdvantages']).items():
            distances[location] = distance_to_meters(distance)
        location_matrix[index] = distances

    # Convert the dictionary to a dataframe
    location_df = pd.DataFrame.from_dict(location_matrix, orient='index')

    # Set the index of 'location_df' to match the 'PropertyName' column in DataFrame 'df'
    location_df.index = df.PropertyName

    # Fill NaN values in 'location_df' with a default value of 54000
    location_df.fillna(54000, inplace=True)

    # Apply the scaler to the entire dataframe
    location_df_normalized = pd.DataFrame(scaler.fit_transform(location_df), columns=location_df.columns, index=location_df.index)

    cosine_sim3 = cosine_similarity(location_df_normalized)

    output_path = home_dir.as_posix() + '/models'

    # Save cosine_sim1 to the specified path
    with open(output_path + '/cosine_sim1.pkl', 'wb') as file:
        pickle.dump(cosine_sim1, file)

    # Save cosine_sim2 to the specified path
    with open(output_path + '/cosine_sim2.pkl', 'wb') as file:
        pickle.dump(cosine_sim2, file)

    # Save cosine_sim3 to the specified path
    with open(output_path + '/cosine_sim3.pkl', 'wb') as file:
        pickle.dump(cosine_sim3, file)


if __name__ == "__main__":
    main()