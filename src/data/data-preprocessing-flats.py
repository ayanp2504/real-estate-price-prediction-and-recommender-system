# data-preprocessing-flats.py
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
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path + '/flats_cleaned.csv', index=False)

def drop_and_rename_cols(df):
    df.drop(columns=['link','property_id'], inplace=True)
    df.rename(columns={'area':'price_per_sqft'},inplace=True)
    return df

def clean_society_col(df):
    df['society'] = df['society'].apply(lambda name: re.sub(r'\d+(\.\d+)?\s?★', '', str(name)).strip()).str.lower()
    return df

def process_price_column(df):
    df = df[df['price'] != 'Price on Request']

    def treat_price(x):
        if type(x) == float:
            return x
        else:
            if x[1] == 'Lac':
                return round(float(x[0]) / 100, 2)
            else:
                return round(float(x[0]), 2)

    df['price'] = df['price'].str.split(' ').apply(treat_price)
    
    return df

def clean_price_per_sqft_col(df):
    df['price_per_sqft'] = df['price_per_sqft'].str.split('/').str.get(0).str.replace('₹','').str.replace(',','').str.strip().astype('float')
    return df

def treat_bedroom_col(df):
    df = df[~df['bedRoom'].isnull()]
    df['bedRoom'] = df['bedRoom'].str.split(' ').str.get(0).astype('int')
    return df

def treat_bathroom_col(df):
    df['bathroom'] = df['bathroom'].str.split(' ').str.get(0).astype('int')
    return df

def treat_balcony_col(df):
    df['balcony'] = df['balcony'].str.split(' ').str.get(0).str.replace('No','0')
    return df

def treat_additional_room_col(df):
    df['additionalRoom'].fillna('not available',inplace=True)
    df['additionalRoom'] = df['additionalRoom'].str.lower()
    return df

def treat_floorNum_col(df):
    df['floorNum'] = df['floorNum'].str.split(' ').str.get(0).replace('Ground','0').str.replace('Basement','-1').str.replace('Lower','0').str.extract(r'(\d+)')
    return df

def treat_facing_col(df):
    df['facing'].fillna('NA',inplace=True)
    return df

def calculate_area_and_insert_col(df):
    df.insert(loc=4,column='area',value=round((df['price']*10000000)/df['price_per_sqft']))
    return df


def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    input_file = sys.argv[1]
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + '/data/processed'
    
    data = load_data(data_path)
    data = drop_and_rename_cols(data)
    data = clean_society_col(data)
    data = process_price_column(data)
    data = clean_price_per_sqft_col(data)
    data = treat_bedroom_col(data)
    data = treat_bathroom_col(data)
    data = treat_balcony_col(data)
    data = treat_additional_room_col(data)
    data = treat_floorNum_col(data)
    data = treat_facing_col(data)
    data = calculate_area_and_insert_col(data)

    data.insert(loc=1,column='property_type',value='flat')
    save_data(data, output_path)

if __name__ == "__main__":
    main()