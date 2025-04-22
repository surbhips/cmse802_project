#gridSearch.py

"""
This module contains grid search for hyperparameters to tune the models. 


Name: Surbhi Punhani Schillinger
Date: 4/8/2025

"""
#import packages
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from src.preprocessing import build_preprocessing_pipeline


def run_grid_search(X, y, model, param_grid, categorical_columns, numerical_columns, cv=5, scoring='r2'):
    """
    Run GridSearchCV on a given model with preprocessing pipeline.

    Parameters
    ----------
    X : pd.DataFrame
        Raw feature data.
    y : pd.Series
        Target variable.
    model : sklearn estimator
        The regression model to tune (e.g., DecisionTreeRegressor()).
    param_grid : dict
        Dictionary of parameters to search, with keys prefixed by 'regressor__'.
    categorical_columns : list of str
        List of categorical feature names.
    numerical_columns : list of str
        List of numerical feature names.
    cv : int, default=5
        Number of cross-validation folds.
    scoring : str, default='r2'
        Scoring metric for GridSearchCV.

    Returns
    -------
    grid_search : GridSearchCV
        The fitted GridSearchCV object.
    """
    # Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(categorical_columns, numerical_columns)

    # Combine with model into a full pipeline
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('regressor', model)
    ])

    # Create and run GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline, #the pipeline of preprocess data and model
        param_grid=param_grid, #dictionary of parameters to search
        cv=cv, #number of cross-validation folds
        scoring=scoring, #scoring metric that is provided which here is r2
        n_jobs=-1, #use all available cores
        return_train_score=True #return training scores
    )

    # Fit the grid search
    grid_search.fit(X, y)

    # Print best parameters and score
    print("Best parameters:", grid_search.best_params_)
    print(f"Best {scoring} score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_

 #help from chatgpt