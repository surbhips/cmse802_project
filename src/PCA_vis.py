#PCA_vis.py

"""
This module contains PCA for the purpose of visualization and compare against linear regression.
I don't think this project needs dimensionality reduction, because the number of features are small. 
But I wanted to add PCA for visualization purposes.


Name: Surbhi Punhani Schillinger
Date: 4/8/2025

"""
#import packages
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from src.preprocessing import build_preprocessing_pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#define function to visualize PCA
def visualize_pca_with_preprocessing(X, categorical_columns=None, numerical_columns=None, n_components=2, color_by=None):
    """
    Preprocess the data, perform PCA, and plot the first 2 principal components.

    Parameters
    ----------
    X : DataFrame
        Raw feature data.
    y : Series, optional
        Target values (used for coloring, if no color_by provided).
    color_by : Series, optional
        A categorical feature for color grouping (e.g., formulation).
    categorical_columns : list of str
        List of categorical columns.
    numerical_columns : list of str
        List of numerical columns.
    abs_column : str
        Column name to apply absolute value transformation.
    n_components : int
        Number of principal components to keep (default is 2).
    """
    # Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(categorical_columns, numerical_columns)

    # Apply preprocessing
    X_processed = preprocessor.fit_transform(X)

    # Run PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_processed)

    # Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=color_by, palette='tab10', s=60, edgecolor='k')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('PCA: First 2 Principal Components')
    plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

 #help from chatgpt