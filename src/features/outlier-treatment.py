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
    data.to_csv(output_path + '/gurgaon_properties_outlier_treated.csv', index=False)


def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    input_file = sys.argv[1]
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + '/data/processed'  
    df = load_data(data_path)

        # Calculate the IQR for the 'price' column
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1

    # Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)]

    #################### Price_per_sqft #################################
    # Calculate the IQR for the 'price' column
    Q1 = df['price_per_sqft'].quantile(0.25)
    Q3 = df['price_per_sqft'].quantile(0.75)
    IQR = Q3 - Q1

    # Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers_sqft = df[(df['price_per_sqft'] < lower_bound) | (df['price_per_sqft'] > upper_bound)]

    outliers_sqft['area'] = outliers_sqft['area'].apply(lambda x:x*9 if x<1000 else x)
    outliers_sqft['price_per_sqft'] = round((outliers_sqft['price']*10000000)/outliers_sqft['area'])
    df.update(outliers_sqft)

    df = df[df['price_per_sqft'] <= 50000]

    #################### Area #################################

    df = df[df['area'] < 100000]
    df.drop(index=[818, 1796, 1123, 2, 2356, 115, 3649, 2503, 1471], inplace=True)
    df.loc[48,'area'] = 115*9
    df.loc[300,'area'] = 7250
    df.loc[2666,'area'] = 5800
    df.loc[1358,'area'] = 2660
    df.loc[3195,'area'] = 2850
    df.loc[2131,'area'] = 1812
    df.loc[3088,'area'] = 2160
    df.loc[3444,'area'] = 1175

     #################### BedRoom #################################
    df = df[df['bedRoom'] <= 10]

    #################### carpet_area #################################
    df.loc[2131,'carpet_area'] = 1812


    df['price_per_sqft'] = round((df['price']*10000000)/df['area'])
    df['area_room_ratio'] = df['area']/df['bedRoom']
    
    save_data(df, output_path)

if __name__ == "__main__":
    main()