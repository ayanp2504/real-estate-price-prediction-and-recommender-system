import pathlib
import sys
import pandas as pd
import numpy as np
import shap
import json
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE

def load_data(data_path):
    # Load your dataset from a given path
    df = pd.read_csv(data_path)
    return df

def save_data(data, output_path):
    # Save the split datasets to the specified output path
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path + '/gurgaon_properties_post_feature_selection.csv', index=False)

def categorize_luxury(score):
    if 0 <= score < 50:
        return "Low"
    elif 50 <= score < 150:
        return "Medium"
    elif 150 <= score <= 175:
        return "High"
    else:
        return None  # or "Undefined" or any other label for scores outside the defined bins

def categorize_floor(floor):
    if 0 <= floor <= 2:
        return "Low Floor"
    elif 3 <= floor <= 10:
        return "Mid Floor"
    elif 11 <= floor <= 51:
        return "High Floor"
    else:
        return None  # or "Undefined" or any other label for floors outside the defined bins
    
def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    input_file = sys.argv[1]
    # input_file = '.\data\processed\gurgaon_properties_missing_value_imputation.csv'
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + '/data/processed'

    
    train_df = load_data(data_path)

    train_df.drop(columns=['society','price_per_sqft'], inplace=True)
    

    ####### luxury score #########
    train_df['luxury_category'] = train_df['luxury_score'].apply(categorize_luxury)

    ####### floorNum #########
    train_df['floor_category'] = train_df['floorNum'].apply(categorize_floor)

    # Drop columns after extracting features
    train_df.drop(columns=['floorNum','luxury_score'],inplace=True)
    

    # Create a copy of the original data for label encoding
    data_label_encoded = train_df.copy()
    
    categorical_cols = train_df.select_dtypes(include=['object']).columns

    # Apply label encoding to categorical columns
    for col in categorical_cols:
        oe = OrdinalEncoder()
        data_label_encoded[col] = oe.fit_transform(data_label_encoded[[col]])

    # Splitting the dataset into training and testing sets
    X_label = data_label_encoded.drop('price', axis=1)
    y_label = data_label_encoded['price']

    ########################### Feature Importance Calculation Techniques #######################

    # Technique 1 - Correlation Analysis
    fi_df1 = data_label_encoded.corr()['price'].iloc[1:].to_frame().reset_index().rename(columns={'index':'feature','price':'corr_coeff'})
    

    # Technique 2 - Random Forest Feature Importance
    # Train a Random Forest regressor on label encoded data
    rf_label = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_label.fit(X_label, y_label)

    # Extract feature importance scores for label encoded data
    fi_df2 = pd.DataFrame({
        'feature': X_label.columns,
        'rf_importance': rf_label.feature_importances_
    }).sort_values(by='rf_importance', ascending=False)

    # Technique 3 - Gradient Boosting Feature importances
    # Train a Random Forest regressor on label encoded data
    gb_label = GradientBoostingRegressor()
    gb_label.fit(X_label, y_label)

    # Extract feature importance scores for label encoded data
    fi_df3 = pd.DataFrame({
        'feature': X_label.columns,
        'gb_importance': gb_label.feature_importances_
    }).sort_values(by='gb_importance', ascending=False)

    # Technique 4 - Permutation Importance

    X_train_label, X_test_label, y_train_label, y_test_label = train_test_split(X_label, y_label, test_size=0.2, random_state=42)

    # Train a Random Forest regressor on label encoded data
    rf_label = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_label.fit(X_train_label, y_train_label)

    # Calculate Permutation Importance
    perm_importance = permutation_importance(rf_label, X_test_label, y_test_label, n_repeats=30, random_state=42)

    # Organize results into a DataFrame
    fi_df4 = pd.DataFrame({
        'feature': X_label.columns,
        'permutation_importance': perm_importance.importances_mean
    }).sort_values(by='permutation_importance', ascending=False)

    

    # Technique 5 - LASSO
        # Train a LASSO regression model
    # We'll use a relatively small value for alpha (the regularization strength)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_label)

    lasso = Lasso(alpha=0.01, random_state=42)
    lasso.fit(X_scaled, y_label)

    # Extract coefficients
    fi_df5 = pd.DataFrame({
        'feature': X_label.columns,
        'lasso_coeff': lasso.coef_
    }).sort_values(by='lasso_coeff', ascending=False)

    # Technique 6 - RFE
        # Initialize the base estimator
    estimator = RandomForestRegressor()

    # Apply RFE on the label-encoded and standardized training data
    selector_label = RFE(estimator, n_features_to_select=X_label.shape[1], step=1)
    selector_label = selector_label.fit(X_label, y_label)

    # Get the selected features based on RFE
    selected_features = X_label.columns[selector_label.support_]

    # Extract the coefficients for the selected features from the underlying linear regression model
    selected_coefficients = selector_label.estimator_.feature_importances_

    # Organize the results into a DataFrame
    fi_df6 = pd.DataFrame({
        'feature': selected_features,
        'rfe_score': selected_coefficients
    }).sort_values(by='rfe_score', ascending=False)

    # Technique 7 - Linear Regression Weights
        # Train a linear regression model on the label-encoded and standardized training data
    lin_reg = LinearRegression()
    lin_reg.fit(X_scaled, y_label)

    # Extract coefficients
    fi_df7 = pd.DataFrame({
        'feature': X_label.columns,
        'reg_coeffs': lin_reg.coef_
    }).sort_values(by='reg_coeffs', ascending=False)

    # Technique 8 - SHAP
    # Compute SHAP values using the trained Random Forest model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_label, y_label)

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_label)

    # Summing the absolute SHAP values across all samples to get an overall measure of feature importance
    shap_sum = np.abs(shap_values).mean(axis=0)
    fi_df8 = pd.DataFrame({
        'feature': X_label.columns,
        'SHAP_score': np.abs(shap_values).mean(axis=0)
    }).sort_values(by='SHAP_score', ascending=False)

    
    # Merging all scores
    final_fi_df = fi_df1.merge(fi_df2,on='feature').merge(fi_df3,on='feature').merge(fi_df4,on='feature').merge(fi_df5,on='feature').merge(fi_df6,on='feature').merge(fi_df7,on='feature').merge(fi_df8,on='feature').set_index('feature')

    # normalize the score
    final_fi_df = final_fi_df.divide(final_fi_df.sum(axis=0), axis=1)

    


    # Features ['pooja room', 'study room', 'others'] scored the least so removing these features.
    export_df = train_df.drop(columns=['pooja room', 'study room', 'others'])
    # Applying the log1p transformation to the target variable
    export_df['price']= np.log1p(train_df['price'])

    save_data(export_df, output_path)



if __name__ == "__main__":
    main()