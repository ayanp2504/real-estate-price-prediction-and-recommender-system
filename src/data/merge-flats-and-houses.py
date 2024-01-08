import pathlib
import sys
import pandas as pd
import numpy as np
import re

def load_data(data_path):
    # Load your dataset from a given path
    df = pd.read_csv(data_path)
    return df

def save_data(data, output_path):
    # Save the split datasets to the specified output path
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path + '/gurgaon_properties.csv', index=False)

def merge_files(file1, file2):
    df = pd.concat([file1,file2],ignore_index=True)
    return df

def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    flat_input_file = sys.argv[1]
    flat_data_path = home_dir.as_posix() + flat_input_file

    house_input_file = sys.argv[2]
    house_data_path = home_dir.as_posix() + house_input_file

    output_path = home_dir.as_posix() + '/data/processed'
    
    flat_data = load_data(flat_data_path)
    house_data = load_data(house_data_path)
    data = merge_files(flat_data, house_data)
    save_data(data, output_path)

if __name__ == "__main__":
    main()