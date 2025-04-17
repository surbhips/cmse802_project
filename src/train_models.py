#train_models.py
"""
This module will be used to train models
This will include all relevant functions and models.

Name: Surbhi Punhani Schillinger
Date: 3/23/2025

"""
#load packages
import importlib
import preprocessing
from preprocessing import load_data
from preprocessing import split_train_test_data
from preprocessing import check_outliers
from preprocessing import build_preprocessing_pipeline
from lin_regress   import train_linear_model_pipeline
from randomForest import train_random_forest_pipeline
from dec_tree import train_decision_tree_pipeline
from svr import train_svr_model_pipeline
from xgb_model import train_xgb_model_pipeline

#load file
X, y = load_data("full_data.csv", "q (crosslink density)")
#drop columns that are not needed
columns_to_drop = ["Temperature (°C)","R (J/K*mol)", "Average density of polymer (g/m^3)", "Mw", "Tan(delta)", "Loss modulus (MPa)", "Sample Number", "Storage modulus (MPa)", "Mc (g/mol)", "Formulation"]
X = X.drop(columns=columns_to_drop)

#get train and test split
X_train, X_test, y_train, y_test = split_train_test_data(X,y)

#split the data into categorical and numerical for both test and train
categorical_columns = ["Polymerization"]
numerical_columns = ["Temp (K)", "per_crosslinker", "dyn_crosslinker", "comonomer"]

X_train = check_outliers(X_train, numerical_columns)
X_train.head()

linReg = train_linear_model_pipeline(X_train, y_train, categorical_columns, numerical_columns, cv = 5)

randomForest = train_random_forest_pipeline(X_train, y_train, categorical_columns, numerical_columns, cv=5)

dec_tree = train_decision_tree_pipeline(X_train, y_train, categorical_columns, numerical_columns, cv=5)

xgboost = train_xgb_model_pipeline(X_train, y_train, categorical_columns, numerical_columns, cv=5)

svr = train_svr_model_pipeline(X_train, y_train, categorical_columns, numerical_columns, cv=5)