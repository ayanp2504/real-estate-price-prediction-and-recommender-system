import pathlib
import sys
import numpy as np
import pandas as pd
import seaborn as sns

def load_data(data_path):
    # Load your dataset from a given path
    df = pd.read_csv(data_path)
    return df

def save_data(data, output_path):
    # Save the split datasets to the specified output path
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path + '/gurgaon_properties_missing_value_imputation.csv', index=False)

def mode_based_imputation(row, df):
    if row['agePossession'] == 'Undefined':
        mode_value = df[(df['sector'] == row['sector']) & (df['property_type'] == row['property_type'])]['agePossession'].mode()
        # If mode_value is empty (no mode found), return NaN, otherwise return the mode
        if not mode_value.empty:
            return mode_value.iloc[0] 
        else:
            return np.nan
    else:
        return row['agePossession']
    
def mode_based_imputation2(row, df):
    if row['agePossession'] == 'Undefined':
        mode_value = df[(df['sector'] == row['sector'])]['agePossession'].mode()
        # If mode_value is empty (no mode found), return NaN, otherwise return the mode
        if not mode_value.empty:
            return mode_value.iloc[0] 
        else:
            return np.nan
    else:
        return row['agePossession']

def mode_based_imputation3(row, df):
    if row['agePossession'] == 'Undefined':
        mode_value = df[(df['property_type'] == row['property_type'])]['agePossession'].mode()
        # If mode_value is empty (no mode found), return NaN, otherwise return the mode
        if not mode_value.empty:
            return mode_value.iloc[0] 
        else:
            return np.nan
    else:
        return row['agePossession']

def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    input_file = sys.argv[1]
    data_path = home_dir.as_posix() + input_file
    df = load_data(data_path)

    all_present_df = df[~((df['super_built_up_area'].isnull()) | (df['built_up_area'].isnull()) | (df['carpet_area'].isnull()))]

    super_to_built_up_ratio = (all_present_df['super_built_up_area']/all_present_df['built_up_area']).median()

    carpet_to_built_up_ratio = (all_present_df['carpet_area']/all_present_df['built_up_area']).median()

        # both present built up null
    sbc_df = df[~(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & ~(df['carpet_area'].isnull())]

    sbc_df['built_up_area'].fillna(round(((sbc_df['super_built_up_area']/1.105) + (sbc_df['carpet_area']/0.9))/2),inplace=True)

    df.update(sbc_df)

    # sb present c is null built up null
    sb_df = df[~(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & (df['carpet_area'].isnull())]

    sb_df['built_up_area'].fillna(round(sb_df['super_built_up_area']/1.105),inplace=True)

    df.update(sb_df)

        # sb null c is present built up null
    c_df = df[(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & ~(df['carpet_area'].isnull())]
    
    c_df['built_up_area'].fillna(round(c_df['carpet_area']/0.9),inplace=True)

    df.update(c_df)

    anamoly_df = df[(df['built_up_area'] < 2000) & (df['price'] > 2.5)][['price','area','built_up_area']]

    anamoly_df['built_up_area'] = anamoly_df['area']

    df.update(anamoly_df)

    df.drop(columns=['area','areaWithType','super_built_up_area','carpet_area','area_room_ratio'],inplace=True)

    ################ floorNum ###########################

    df[df['property_type'] == 'house']['floorNum'].median()
    df['floorNum'].fillna(2.0,inplace=True)

    ################# facing #######################

    df.drop(columns=['facing'],inplace=True)
    df.drop(index=[2536],inplace=True)

    ################# agePossession #################
    df[df['agePossession'] == 'Undefined']
    df['agePossession'] = df.apply(mode_based_imputation, args=(df,), axis=1)
    df['agePossession'] = df.apply(mode_based_imputation2,args=(df,), axis=1)
    df['agePossession'] = df.apply(mode_based_imputation3,args=(df,), axis=1)



    output_path = home_dir.as_posix() + '/data/processed'
    save_data(df, output_path)

if __name__ == "__main__":
    main()