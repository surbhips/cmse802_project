#PCA_vis.py

"""
This module contains Decision Tree for training this model. 


Name: Surbhi Punhani Schillinger
Date: 4/8/2025

"""
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, root_mean_squared_error, mean_absolute_error
from preprocessing import build_preprocessing_pipeline

def train_decision_tree_pipeline(X, y, categorical_columns, numerical_columns, cv=5):
    """
    Train a Decision Tree Regressor model using a full preprocessing + modeling pipeline with cross-validation.

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
    #get model
    model = DecisionTreeRegressor(random_state=42)

    # Preprocessing block
    preprocessor = build_preprocessing_pipeline(categorical_columns, numerical_columns)

    # Create pipeline with preprocessing and model
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('regressor_decTree', model)
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
    print("Decision Tree Cross-Validation Scores:")
    print(f"  R²:  {scores['test_r2'].mean():.4f} ± {scores['test_r2'].std():.4f}")
    print(f"  RMSE: {scores['test_rmse'].mean():.4f} ± {scores['test_rmse'].std():.4f}")
    print(f"  MAE: {scores['test_mae'].mean():.4f} ± {scores['test_mae'].std():.4f}")

    return scores