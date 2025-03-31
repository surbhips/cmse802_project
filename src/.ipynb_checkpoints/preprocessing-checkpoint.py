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
from preprocessing import load_data
from preprocessing import split_train_test_data

#load file
def load_data(file):
    """
    load file
    
    Parameters
    ----------
    data : pandas.DataFrame
        Raw input data with features as columns
   X, y : get the data splot 
        
    Returns X and y
    -------
    pandas.DataFrame
        Converted data ready for next steps of preprocesing
    """
    df = pd.read_csv(file)
    X = df.drop("target", axis = 1)
    y = df["target"]
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
    return train_test_split(X, y, test_size=test_size, random_state=random_state)



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


#encode the categorical data
def fit_one_hot_encoder(df, categorical_data):
    """
    Fit one-hot encoder to training dataset.
    
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
    fit_enc = OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore')
    fit_enc.fit(df[categorical_data])
    return fit_enc

#function to transform the data
def transform_one_hot_encoder(df, encoder, categorical_data):
    """
    Transform categorical data using a fitted OneHotEncoder.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing categorical columns.
    encoder : OneHotEncoder
        Fitted OneHotEncoder object.
    categorical_data : list of str
        List of categorical column names to transform.
    
    Returns
    -------
    pandas.DataFrame
        A new DataFrame with original categorical columns replaced by one-hot encoded columns.
    
    Examples
    --------
    >>> df = pd.read_csv('full_data.csv')
    >>> encoder = fit_one_hot_encoder(df, ['polymerization'])
    >>> df_transformed = transform_one_hot_encoder(df, encoder, ['polymerization'])
    """
    transformed = encoder.transform(df[categorical_data])
    encoded_df = pd.DataFrame(
        transformed,
        columns=encoder.get_feature_names_out(categorical_data),
        index=df.index
    )
    df = df.drop(columns=categorical_data)
    return pd.concat([df, encoded_df], axis=1)



    
