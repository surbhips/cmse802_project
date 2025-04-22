#Unit tests
"""
This module is for testing all the functions used for preprocessing.

This contains unit tests to check if the functions like absolute value conversion, 
outlier detection,scaling and standardizing and converting categorical data.


Name: Surbhi Punhani Schillinger
Date: 3/21/2025

"""
### Test case Design
import pytest
import pandas as pd
import numpy as np
from io import StringIO
from src.preprocessing import load_data, split_train_test_data, check_outliers, build_preprocessing_pipeline

# Sample CSV content for testing `load_data`
csv_content = """feature1,feature2,category,target
1,10,A,0
2,15,B,1
3,20,C,0
4,25,A,1
"""

# Create a temporary CSV file fixture to use across tests
@pytest.fixture
def mock_csv_file(tmp_path):
    file = tmp_path / "data.csv"
    file.write_text(csv_content)
    return str(file)

# Test load_data returns correct structure and splits columns
def test_load_data_valid(mock_csv_file):
    X, y = load_data(mock_csv_file, target_column='target')
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert 'target' not in X.columns  # Target column should be excluded from X
    assert len(X) == len(y) == 4

# Test load_data raises an error when the target column is missing
def test_load_data_missing_target(mock_csv_file):
    with pytest.raises(KeyError):
        load_data(mock_csv_file, target_column='not_a_column')

# Test train/test splitting returns correct sizes and types
def test_split_train_test_data_valid():
    X = pd.DataFrame({'a': range(10)})
    y = pd.Series(range(10))
    X_train, X_test, y_train, y_test = split_train_test_data(X, y, test_size=0.3, random_state=0)
    assert len(X_train) == 7
    assert len(X_test) == 3
    assert all(isinstance(df, (pd.DataFrame, pd.Series)) for df in [X_train, X_test, y_train, y_test])

# Test splitting on empty data raises an error
def test_split_train_test_data_empty():
    X = pd.DataFrame()
    y = pd.Series(dtype=int)
    with pytest.raises(ValueError):
        split_train_test_data(X, y)

# Test check_outliers when no outliers are present (should return full data)
def test_check_outliers_no_outliers():
    df = pd.DataFrame({'a': [10, 12, 14, 13, 11]})
    result = check_outliers(df, ['a'])
    assert len(result) == 5  # All data points remain

# Test check_outliers when there is an extreme value (should be removed)
def test_check_outliers_with_outliers():
    df = pd.DataFrame({'a': [10, 12, 14, 13, 11, 1000]})
    result = check_outliers(df, ['a'])
    assert 1000 not in result['a'].values  # Outlier should be excluded
    assert len(result) == 5

# Test check_outliers with an empty DataFrame
def test_check_outliers_empty_df():
    df = pd.DataFrame(columns=['a'])
    result = check_outliers(df, ['a'])
    assert result.empty  # Should return an empty DataFrame without error

# Test the preprocessing pipeline builds and transforms data as expected
def test_build_preprocessing_pipeline_runs():
    df = pd.DataFrame({
        'cat': ['A', 'B', 'C', 'A'],
        'num1': [1.0, 2.0, 3.0, 4.0],
        'num2': [10.0, 20.0, 30.0, 40.0]
    })
    # Build pipeline with 1 categorical and 2 numerical features
    pipe = build_preprocessing_pipeline(categorical_columns=['cat'], numerical_columns=['num1', 'num2'])
    transformed = pipe.fit_transform(df)

    # Check output is a NumPy array and has correct number of rows
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == 4



#help from: chatgpt