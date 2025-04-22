#lin_regress.py

"""
This module contains linear regression model that will be used to train the dataset. 


Name: Surbhi Punhani Schillinger
Date: 4/8/2025

"""
#import packages
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, root_mean_squared_error, mean_absolute_error
from src.preprocessing import build_preprocessing_pipeline

def train_linear_model_pipeline(X, y, categorical_columns, numerical_columns, cv=5):
    """
    Train a Linear Regression model using a full preprocessing + modeling pipeline.

    Parameters
    ----------
    X : pd.DataFrame
        Raw feature data (not preprocessed).
    y : pd.Series
        Target variable.
    categorical_columns : list of str
        Categorical columns to one-hot encode.
    numerical_columns : list of str
        Numerical columns to standardize.
    cv : int
        Number of cross-validation folds.

    Returns
    -------
    scores : dict
        Dictionary with cross-validation R², MSE, and MAE scores.
    """

    # Preprocessing block
    preprocessor = build_preprocessing_pipeline(categorical_columns, numerical_columns)

    # Build pipeline steps
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('regressor_lin', LinearRegression())
    ])

    # Scoring metrics
    scoring = {
        'r2': 'r2',
        'rmse': make_scorer(root_mean_squared_error),
        'mae': make_scorer(mean_absolute_error)
    }

    # Cross-validation
    scores = cross_validate(pipeline, X, y, scoring=scoring, cv=cv, return_train_score=False)

    print("Linear Regression Cross-Validation Scores:")
    print(f"  R^2:  {scores['test_r2'].mean():.4f} ± {scores['test_r2'].std():.4f}")
    print(f"  RMSE:  {scores['test_rmse'].mean():.4f} ± {scores['test_rmse'].std():.4f}")
    print(f"  MAE:  {scores['test_mae'].mean():.4f} ± {scores['test_mae'].std():.4f}")

    return scores

 #help from chatgpt