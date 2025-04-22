#randomForest.py

"""
This module contains random forest model that will be used to train the dataset. 


Name: Surbhi Punhani Schillinger
Date: 3/23/2025

"""
#import packages
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, root_mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from src.preprocessing import build_preprocessing_pipeline

#function to train the random forest model
def train_random_forest_pipeline(X, y, categorical_columns, numerical_columns, cv=5):
    """
    Train a Random Forest model using a full preprocessing + modeling pipeline with cross-validation.

    Parameters
    ----------
    X : pandas.DataFrame
        Raw feature data (not preprocessed).
    y : pandas.Series
        Target variable.
    categorical_columns : list of str
        Categorical columns to one-hot encode.
    numerical_columns : list of str
        Numerical columns to standardize.
    abs_column : str
        Column name to apply absolute value transformation.
    cv : int
        Number of cross-validation folds.

    Returns
    -------
    scores : dict
        Dictionary of cross-validated R², MSE, and MAE scores.
    """
    #define the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Preprocessing block
    preprocessor = build_preprocessing_pipeline(categorical_columns, numerical_columns)

    # Create pipeline with preprocessing and model
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('regressor_rf', model)
    ])

    # Scoring metrics
    scoring = {
        'r2': 'r2',
        'rmse': make_scorer(root_mean_squared_error),
        'mae': make_scorer(mean_absolute_error)
    }

    # Perform cross-validation
    scores = cross_validate(pipeline, X, y, scoring=scoring, cv=cv, return_train_score=False)

    # Print results
    print("Random Forest Cross-Validation Scores:")
    print(f"  R²:  {scores['test_r2'].mean():.4f} ± {scores['test_r2'].std():.4f}")
    print(f"  RMSE: {scores['test_rmse'].mean():.4f} ± {scores['test_rmse'].std():.4f}")
    print(f"  MAE: {scores['test_mae'].mean():.4f} ± {scores['test_mae'].std():.4f}")

    return scores

 #help from chatgpt