#Unit tests
"""
This module is for testing all the functions used for preprocessing.

This contains unit tests to check if the functions like absolute value conversion, 
outlier detection,scaling and standardizing and converting categorical data.


Name: Surbhi Punhani Schillinger
Date: 3/21/2025

"""
### Test case Design
import unittest
import numpy as np
import pandas as pd

class TestPreprocessingFunctions(unittest.TestCase):
    def unit_test(self):
          """
        Functionality:
        Test load file works
        
        Input:  df = pd.read_csv(data.csv)
            X = df.drop[target]
            y = df[target]

        Expected output: df

        Validation:
        Check to see if the function loads the file and creates X and y
        """
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".csv") as tmp:
        tmp.write("feature1,feature2,target\n")
        tmp.write("1,2,0\n")
        tmp.write("3,4,1\n")
        tmp_path = tmp.name

    # Call the function
    X, y = load_data(tmp_path)

    # Clean up temp file
    os.remove(tmp_path)

    # Check shapes and content
    assert X.shape == (2, 2)
    assert y.shape == (2,)
    assert list(X.columns) == ["feature1", "feature2"]
    assert y.tolist() == [0, 1]

    #unit test for splitting train and test data
    def test_split_train_test_data():
          """
        Functionality:
        Test split trian and test data
        
        Input:  X and y
           

        Expected output: X_train, y_train, X_test, y_test

        Validation:
        Check to see if the function returns a split dataset
        """
    # Sample dummy dataset
    X = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 20, 30, 40, 50]
    })
    y = pd.Series([0, 1, 0, 1, 0])

    X_train, X_test, y_train, y_test = split_train_test_data(X, y, test_size=0.4, random_state=0)

    # Check lengths
    assert len(X_train) == 3
    assert len(X_test) == 2
    assert len(y_train) == 3
    assert len(y_test) == 2

    # Check alignment of train/test
    assert X_train.index.equals(y_train.index)
    assert X_test.index.equals(y_test.index)
    
    def test_abs(self):
        """
        Functionality:
        Test if the function test_abs actually converts a column into it's absolute value to get rid of any negatives
        
        Input:  df = pd.DataFrame({
            'A': [-1, 0, 5, -10]
        })

        Expected output: expected = pd.series([1, 0 5, 10], name=A)

        Validation:
        Check to see if the result is what is expected of the function. The result should be pass or fail
        """

        df = pd.DataFrame({
            'A': [-1, 0, 5, -10]
        })

        expected = pd.series([1, 0 5, 10], name=A)

        result = to_abs(df.copy(), 'A')

        pd.testing.assert_series_equal(result['A'], expected)

    def test_fit_and_transform_one_hot_encoder(self):
         """
        Functionality:
        Test if the function fit and transform columns using one hot encoder and return dataframe with new columns
        
        Input:  df = pd.DataFrame({
            'A': ['2', '3', '5']
            'B': ['a', 'b', 'c']
        })

        Expected output: expected = pd.dataframe

        Validation:
        Check to see if the one hot encoder works with the function. The result should be pass or fail
        """
        df = pd.DataFrame({
            'Curing_Method': ['254, '365, '254', '365'],
            'Formulation': ['0.5:0.5:99', '1:1:98', '0.5:4.5:95']
        })
        categorical_data = ['Curing_Method']

        # Fit encoder on training data
        encoder = fit_one_hot_encoder(df, categorical_data)

        # Transform the same data
        transformed_df = transform_with_encoder(df, encoder, categorical_data)

        # Ensure original column is dropped and new columns are added
        self.assertNotIn('Curing_Method', transformed_df.columns)
        self.assertTrue(any("Curing_Method" in col for col in transformed_df.columns))
        self.assertEqual(len(transformed_df), len(df))






    if __name__ == '__main__':
            unittest.main()