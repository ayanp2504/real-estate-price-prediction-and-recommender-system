import pandas as pd
import sys
import ast
import pickle
import pathlib
from wordcloud import WordCloud

def load_data(data_path):
    # Load your dataset from a given path
    df = pd.read_csv(data_path)
    return df

def save_data(data, output_path, file_name):
    # Save the split datasets to the specified output path
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path + file_name, index=False)

def sep_lat_long(latlong):
    '''This function coordinates into its latitude and longitude'''
    latlong['latitude'] = latlong['coordinates'].str.split(',').str.get(0).str.split('°').str.get(0).astype('float')
    latlong['longitude'] = latlong['coordinates'].str.split(',').str.get(1).str.split('°').str.get(0).astype('float')
    return latlong

def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    input_file1 = sys.argv[1]
    data_path1 = home_dir.as_posix() + input_file1
    latlong = load_data(data_path1)
    latlong = sep_lat_long(latlong)

    # Merge latlong data with main dataset
    input_file2 = sys.argv[2]
    data_path2 = home_dir.as_posix() + input_file2
    df = load_data(data_path2)
    new_df = df.merge(latlong, on='sector')
    output_path = home_dir.as_posix() + '/data/processed'
    save_data(new_df, output_path, '/data_viz1.csv')

    input_file3= sys.argv[3]
    data_path3 = home_dir.as_posix() + input_file3
    df1 = load_data(data_path3)
    wordcloud_df = df1.merge(df, left_index=True, right_index=True)[['features','sector']]
    main = []
    for item in wordcloud_df['features'].dropna().apply(ast.literal_eval):
        main.extend(item)
    feature_text = ' '.join(main)

    file_path = home_dir.as_posix() + '/models/feature_text.pkl'
    
    # Dump the variable to the specified path
    with open(file_path, 'wb') as file:
        pickle.dump(feature_text, file)


if __name__ == "__main__":
    main()