# mfc/mfc/data_generation.py

import numpy as np
import pandas as pd

def load_time_series_data(filepath: str) -> Dict[str, List[int]]:
    """
    Loads time series data from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        dict: Dictionary with CA IDs as keys and lists of state values as values.
    """
    df = pd.read_csv(filepath)
    data = {}
    for column in df.columns:
        data[column] = df[column].astype(int).tolist()
    return data
