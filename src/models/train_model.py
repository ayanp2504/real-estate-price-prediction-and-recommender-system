import sys
import pathlib
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

def load_data(data_path):
    # Load your dataset from a given path
    df = pd.read_csv(data_path)
    return df

def save_model(model, output_path):
    # Save the split datasets to the specified output path
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    model_path = output_path + '/trained_model.pkl'
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)


def train_model(df):

    X = df.drop(columns=['price'])
    y = df['price']

    columns_to_encode = ['property_type','sector', 'balcony', 'agePossession', 'furnishing_type', 'luxury_category', 'floor_category']

    # Preprocess the data
    preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']),
        ('cat', OrdinalEncoder(), columns_to_encode),
        ('cat1',OneHotEncoder(drop='first',sparse=False),['sector','agePossession'])
    ], 
    remainder='passthrough'
    )

        # Set the hyperparameters for RandomForestRegressor
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
    pipeline.fit(X,y)

    return pipeline

def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    input_file = sys.argv[1]
    data_path = home_dir.as_posix() + input_file
    df = load_data(data_path)
    model = train_model(df)

    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    output_path = home_dir.as_posix() + '/models'
    save_model(model, output_path)


if __name__ == "__main__":
    main()