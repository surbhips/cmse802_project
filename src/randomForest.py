#randomForest.py

"""
This module contains random forest model that will be used to train the dataset. 


Name: Surbhi Punhani Schillinger
Date: 3/23/2025

"""
#import packages
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

#create a train function that takes in dataframe
def train(X_train, y_train, cv = 5):
    """
    Train a Random Forest model on the provided training data.
    
    Parameters
    ----------
    X_train : pandas.DataFrame
        The training feature data.
    y_train : pandas.Series
        The training target data.
    """

    #process the dataset into the model to fit
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring ='r2')
    print(f"Cross-Validation R^2 Scores: {scores}")
    return scores


#help from: chatgpt
#not quite done yet