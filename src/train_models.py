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
from preprocessing import fit_trans_one_hot_encoder
from preprocessing import transform_one_hot_encoder
from preprocessing import to_abs
from preprocessing import split_cat_num_data
from preprocessing import fit_transform_standardized, transform_standardized
from randomForest import train
importlib.reload(preprocessing)

#load data
X, y = load_data("full_data.csv", target_column = "q (crosslink density)")
#drop columns that are not needed
columns_to_drop = ["Temperature (°C)","R (J/K*mol)", "Average density of polymer (g/m^3)", "Mw"]
X = X.drop(columns=columns_to_drop)
X.head()

#split the data into train and test
X_train, X_test, y_train, y_test = split_train_test_data(X, y, test_size=0.2, random_state=42)



#split the data into categorical and numerical for both test and train
categorical_columns = ['Formulation', 'Polymerization', 'Sample Number', 'Temp (K)']
numerical_columns = ['Tan(delta)', 'Storage modulus (MPa)', 'Loss modulus (MPa)', 'Mc (g/mol)']
categorical_train_df, numerical_train_df = split_cat_num_data(X_train, categorical_columns)
categorical_test_df, numerical_test_df = split_cat_num_data(X_test, categorical_columns)
categorical_test_df.head()

#fit and transform categorical data for training
fit_trans_trainSet, encoder = fit_trans_one_hot_encoder(X_train, categorical_columns)
trans_testSet = transform_one_hot_encoder(X_test, encoder, categorical_columns)

#convert the Mc to absolute values because it is the molecular weight between two crosslinks physically it is positive
#due to the way it is calculated, sometimes it is negative
X_train_abs = to_abs(fit_trans_trainSet, column_name=["Mc (g/mol)"])
X_test_abs = to_abs(trans_testSet, column_name=["Mc (g/mol)"])

#standardize the numerical data
X_train_num, scaler = fit_transform_standardized(X_train_abs, numerical_columns)
X_test_num = transform_standardized(X_test_abs, numerical_columns, scaler)

#redefine x_train and x_test
X_train = X_train_num
X_test = X_test_num

randomForest = train(X_train, y_train, cv = 5)

if __name__ == "__main__":
    print(process_data())

