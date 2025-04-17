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
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Load and split data
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
    # Load file
    df = pd.read_csv(file)
    # Drop the target column to get X
    X = df.drop(columns=[target_column])
    # Get y
    y = df[target_column]
    return X, y


def split_train_test_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Outlier detection function using IQR
def check_outliers(df, numerical_columns):
    df_clean = df.copy()
    """
    Identify and visualize outliers in numerical columns using IQR method.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataset.
    numerical_columns : list of str
        The list of numerical columns to check.

    Returns
    -------
    outliers : dict
        Dictionary where keys are column names and values are DataFrames with outliers.
    """
    #empty dictionary to store outliers
    outliers = {}
    #check for Q1 and Q3 and IQR
    for col in numerical_columns:
        #check for the first and third quartile
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        #check for the interquartile range
        IQR = Q3 - Q1
        #check for the lower and upper bound for the outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        #return df without the outliers
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean


# Preprocessing pipeline builder
def build_preprocessing_pipeline(categorical_columns, numerical_columns):
    """
    Build a preprocessing pipeline for categorical and numerical data.
    Parameters
    ----------
    categorical_columns : list of str
        List of categorical column names.
        numerical_columns : list of str
        List of numerical column names.
        abs_column : str
            
        Column name to apply absolute value transformation.
    Returns 
    -------
        preprocessor : ColumnTransformer
        Preprocessing pipeline.
    """
    #process for different columns
    preprocessor = ColumnTransformer(
        transformers=[
            #apply one hot encoder to categorical data
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_columns),
            #apply absolute to the column and the standard scaler to the numerical data
            ('num', Pipeline([
                #transform numerical data to absoulte value and scale it
                ('scaler', StandardScaler())
            ]), numerical_columns)
        ]
    )
    #return the preprocessed data
    return preprocessor

#help from: chatgpt