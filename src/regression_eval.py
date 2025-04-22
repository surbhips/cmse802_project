#regression_eval.py
"""
This module will be used to get the evaluation on the test dataset.

Name: Surbhi Punhani Schillinger
Date: 4/16/2025

"""
from sklearn.metrics import r2_score, make_scorer, root_mean_squared_error, mean_absolute_error

#function for getting the scores
#for evaluating the function with test dataset
def evaluate_regression(y_true, y_pred, model_name="Model"):
    """
    Print and return R², RMSE, and MAE for the regression model.
    
    Parameters
    ----------
    y_true : array-like
        Ground truth target values.
    y_pred : array-like
        Predicted values from model.
    model_name : str
        Optional name for the model.
    
    Returns
    -------
    dict
        Dictionary containing the scores.
    """
    #get the three scores based on the test q and the predicted q
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)

    #print the scores for the model for the evaluation
    print(f" Evaluation Metrics for {model_name}:")
    print(f"  R²   : {r2:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")

    #retun the model name, the R^2, RMSE and MAE
    return {'model': model_name, 'r2': r2, 'rmse': rmse, 'mae': mae}

 #help from chatgpt