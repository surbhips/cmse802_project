#feat_eng.py
"""
This module will be used to apply feature engineering


Name: Surbhi Punhani Schillinger
Date: 4/16/2025

"""
import pandas as pd
def apply_feature_engineering(df):
    """
    Adds engineered features:
    - Percent crosslinker and monomer
    - Ratio of UV to thermal crosslinker

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'uv_crosslinker', 'thermal_crosslinker', 'base_monomer'

    Returns
    -------
    df : pd.DataFrame
        Updated DataFrame with new features added
    """
    # Convert raw values to float
    df['dyn_crosslinker'] = df['dyn_crosslinker'].astype(float) 
    df['per_crosslinker'] = df['per_crosslinker'].astype(float)
    df['comonomer'] = df['comonomer'].astype(float)

    # Total sum of components
    #this is to ensure that the percentages are not more than 100%
    total = df['dyn_crosslinker'] + df['per_crosslinker'] + df['comonomer']

    # Calcultte percentages
    #the inputs are in percentage but not in decimal, meaning not the best way to represent the percentage
    df['dyn_crosslinker_pct'] = df['dyn_crosslinker'] / total #dynamic crosslinker percentage
    df['per_crosslinker_pct'] = df['per_crosslinker'] / total #permanent crosslinker percentage
    df['comonomer_pct'] = df['comonomer'] / total #comonomer percentage

    # Ratio feature
    #adding a small tolerance so it never divides by 0, not a current inssure in this data but for future this could be useful
    df['crosslink_ratio'] = df['dyn_crosslinker_pct'] / (df['per_crosslinker_pct'] + 1e-6)

    #return the df
    return df

     #help from chatgpt