from sklearn.datasets import fetch_california_housing
import pandas as pd


def load_housing_data():
    """
    Loads the California Housing dataset.
    Returns:
        pd.DataFrame: The California Housing dataset as a DataFrame.
    """
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    print("First 5 rows of the dataset:")
    print(df.head())
    return df
