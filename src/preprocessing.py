#preprocessing data
"""
This module is for functions that will be used to preprocess data. Before it is trained.

This contains functions for converting them into their absolute function if necessary. Scaling and normalizing the data for model training.
Converting categorical data into encoders.
Removing outliers from the data.

Name: Surbhi Punhani Schillinger
Date: 3/21/2025

"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#load file
import pandas as pd

def load_data(file, target_column):
    """
    Load a CSV file and split it into features (X) and target (y).
    
    Parameters
    ----------
    file : str
        Path to the CSV file.
    
    Returns
    -------
    X : pandas.DataFrame
        Feature data (all columns except 'target').
    
    y : pandas.Series
        Target data (the 'target' column).
    
    Example
    -------
    >>> X, y = load_data("dataset.csv")
    """
    #load file
    df = pd.read_csv(file)
    #drop the target column to get X
    X = df.drop(columns=[target_column])
    #define y as the target column
    y = df[target_column]
    return X, y



#function to split train and test data
def split_train_test_data(X, y, test_size=0.2, random_state=42):
    """
    Splits features and labels into training and testing sets.

    Parameters
    ----------
    data: X and y

    Returns X_train, y_train, X_test, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

#function to split the numerical and categorical data
def split_cat_num_data(df, categorical_columns):
    """
    Split X into categorical and numerical data- this should be ran twice once for training and other for testing
    
    Parameters
    ----------
    df : pandas.DataFrame
        X dataset
    categorical_columns: list of strings
        Columns like "polymerization"
    
    Returns
    -------
    categorical_data : list of str
        Column names that are categorical.
    numerical_data : list of numerical
        Column names that are numerical.
    
    Examples
    --------
    >>> cat_df, num_df = split_cat_num_data(X_train, ["polymerization"])
    """
    # Get categorical df
    categorical_df = df[categorical_columns]
    # The numerical columns are the columns that are not in categorical columns
    numerical_columns = [col for col in df.columns if col not in categorical_columns]
    # Get new df for numerical data
    numerical_df = df[numerical_columns]
    return categorical_df, numerical_df


#function to convert a column into it's absolute value
def to_abs(df, column_name):
    """
    Converts all values in the specified column of a DataFrame to absolute values.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to convert.
        
    Returns:
        pd.DataFrame: A new DataFrame with the specified column converted to absolute values.
    """
    df[column_name] = df[column_name].abs()
    return df




#For linear regression and SVM the data should be standardized
def fit_transform_standardized(df, numerical_columns):
    """
    Fit and transform numerical training data using StandardScaler.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing numerical columns.

    numerical_columns : list of str
        List of column names to standardize.
    
    Returns
    -------
    scaled_df : pandas.DataFrame
        Standardized numerical columns.
    
    scaler : StandardScaler
        Fitted scaler object.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[numerical_columns])
    scaled_df = pd.DataFrame(scaled, columns=numerical_columns, index=df.index)
    return scaled_df, scaler

def transform_standardized(df, numerical_columns, scaler):
    """
    Transform numerical testing data to standard scaler.
    
    Parameters
    ---------- 
    df : pandas.DataFrame
        Input data containing numerical columns.

    numerical_columns : list of numerical data like tan_delta, loss modulus etc.
        List of testing numerical column names to fit.
    
    scaler : StandardScaler
        Fitted scaler object.
    
    Returns
    -------
    scaled df
        Data columns that are transformed to the scaler.
    
    Examples
    --------
    >>> df = pd.read_csv('full_data.csv')
    >>> num_data = transform(df, numerical_data)
    return new_df
    
    """
    scaled_test = scaler.transform(df[numerical_columns])
    scaled_df = pd.DataFrame(scaled_test, columns=numerical_columns, index=df.index)
    return scaled_df

#encode the categorical data
def fit_trans_one_hot_encoder(df, categorical_data):
    """
    Fit and transform one-hot encoder to training dataset.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Raw input data with features as columns.
    categorical_data : list of str
        Column names that are categorical.
    
    Returns
    -------
    OneHotEncoder
        Fitted encoder that can be used to transform other data.
    
    Examples
    --------
    >>> df = pd.read_csv('full_data.csv')
    >>> encoder = fit_one_hot_encoder(df, ['polymerization'])
    >>> transformed = encoder.transform(df[['polymerization']])
    """
    #call encoder
    encoder = OneHotEncoder(sparse_output = False, drop='first', handle_unknown='ignore')
    #fit and transform the data
    encoded_train = encoder.fit_transform(df[categorical_data])
    #create new df with the fit and transformed data
    encoded_df = pd.DataFrame(encoded_train, columns = encoder.get_feature_names_out(categorical_data), index=df.index)
    #drop the original data and get a new df with the new data
    df = df.drop(columns=categorical_data)
    fit_trans_df = pd.concat([df, encoded_df], axis=1)
    return fit_trans_df, encoder

#function to transform the data
def transform_one_hot_encoder(df, encoder, categorical_data):
    """
    Transform categorical data for the test set.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing categorical columns.
    encoder : OneHotEncoder
        
    categorical_data : list of str
        List of test categorical column names to transform.
    
    Returns
    -------
    pandas.DataFrame
        A new DataFrame with original categorical columns replaced by one-hot encoded columns.
    
    Examples
    --------
    >>> df = pd.read_csv('full_data.csv')
    >>> encoder = fit_trans_one_hot_encoder(df, ['polymerization'])
    >>> df_transformed = transform_one_hot_encoder(df, encoder, ['polymerization'])
    """
    #transform the data using fitted encoder
    transformed = encoder.transform(df[categorical_data])
    #get new columns
    encoded_df = pd.DataFrame(
        transformed,
        columns=encoder.get_feature_names_out(categorical_data),
        index=df.index
    )
    #create new df that drops the old data
    df = df.drop(columns=categorical_data)
    #concatenate the new columns, axis = 1 means column wise
    transformed_df = pd.concat([df, encoded_df], axis=1)
    return transformed_df

    
#help from: chatgpt