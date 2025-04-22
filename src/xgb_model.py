#xgb_model.py

"""
This module contains code for xgboost to train the data. 


Name: Surbhi Punhani Schillinger
Date: 4/15/2025

"""


from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import r2_score, make_scorer, root_mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from src.preprocessing import build_preprocessing_pipeline


def train_xgb_model_pipeline(X, y, categorical_columns, numerical_columns, cv=5):
    """
    Train an XGBoost Regression model with preprocessing and cross-validation.
    
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

    preprocessor = build_preprocessing_pipeline(categorical_columns, numerical_columns)

    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('regressor_xgb', XGBRegressor(n_estimators=100, random_state=42))
    ])

    scoring = {
        'r2': 'r2',
        'rmse': make_scorer(root_mean_squared_error),
        'mae': make_scorer(mean_absolute_error)
    }

    scores = cross_validate(pipeline, X, y, scoring=scoring, cv=cv, return_train_score=False)

    print("XGBoost Cross-Validation Scores:")
    print(f"  R²:   {scores['test_r2'].mean():.4f} ± {scores['test_r2'].std():.4f}")
    print(f"  RMSE:  {scores['test_rmse'].mean():.4f} ± {scores['test_rmse'].std():.4f}")
    print(f"  MAE:  {scores['test_mae'].mean():.4f} ± {scores['test_mae'].std():.4f}")

    return scores

 #help from chatgpt