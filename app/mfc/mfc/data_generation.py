# File: mfc/mfc/data_generation.py

import numpy as np
import pandas as pd

def load_time_series_data(filepath: str):
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

def generate_synthetic_time_series_data(num_series, series_length, num_anomalies, anomaly_magnitude, seed=None):
    """
    Generates synthetic time series data with anomalies for testing.

    Args:
        num_series (int): Number of time series to generate.
        series_length (int): Length of each time series.
        num_anomalies (int): Number of anomalies to introduce per series.
        anomaly_magnitude (float): Magnitude of the anomalies.
        seed (int): Random seed for reproducibility.

    Returns:
        pandas.DataFrame: DataFrame containing the generated time series.
    """
    if seed is not None:
        np.random.seed(seed)

    data = {}
    for i in range(num_series):
        # Generate a base time series with some trends and seasonality
        base_series = np.linspace(0, 10, series_length) + np.sin(np.linspace(0, 5 * np.pi, series_length)) * 5
        base_series += np.random.normal(0, 1, series_length)  # Add some noise

        # Introduce anomalies
        for _ in range(num_anomalies):
            anomaly_start = np.random.randint(0, series_length - 1)
            anomaly_duration = np.random.randint(5, 15)  # Vary anomaly duration
            anomaly_end = min(anomaly_start + anomaly_duration, series_length)

            # Different types of anomalies
            anomaly_type = np.random.choice(['spike', 'dip', 'shift'])
            if anomaly_type == 'spike':
                base_series[anomaly_start:anomaly_end] += anomaly_magnitude
            elif anomaly_type == 'dip':
                base_series[anomaly_start:anomaly_end] -= anomaly_magnitude
            elif anomaly_type == 'shift':
                base_series[anomaly_start:] += anomaly_magnitude

        data[f'CA{i + 1}'] = base_series

    return pd.DataFrame(data)

def create_anomalous_data(data: pd.DataFrame, anomalies: list) -> pd.DataFrame:
    """
    Introduces anomalies into the provided time series data based on specified anomaly types.

    Args:
        data (pd.DataFrame): The original time series data.
        anomalies (list): A list of dictionaries, each defining an anomaly with type, start, and magnitude.

    Returns:
        pd.DataFrame: Time series data with anomalies introduced.
    """
    df = data.copy()
    for anomaly in anomalies:
        anomaly_type = anomaly['type']
        start_index = anomaly['start']
        magnitude = anomaly['magnitude']

        if anomaly_type in ['spike', 'dip', 'shift']:
            duration = anomaly.get('duration', 10)  # Default duration if not specified
            end_index = min(start_index + duration, len(df))

            for column in df.columns:
                if anomaly_type == 'spike':
                    df.loc[start_index:end_index, column] += magnitude
                elif anomaly_type == 'dip':
                    df.loc[start_index:end_index, column] -= magnitude
                elif anomaly_type == 'shift':
                    df.loc[start_index:end_index, column] += magnitude
        elif anomaly_type == 'oscillation':
            oscillation_amplitude = anomaly.get('amplitude', 5)
            oscillation_frequency = anomaly.get('frequency', 0.1)
            oscillation_duration = anomaly.get('duration', 20)
            for column in df.columns:
                for i in range(start_index, min(start_index + oscillation_duration, len(df))):
                    df.loc[i, column] += oscillation_amplitude * np.sin(2 * np.pi * oscillation_frequency * i)
        elif anomaly_type == 'trend_change':
            trend_duration = anomaly.get('duration', 30)
            trend_slope = anomaly.get('slope', 0.5)
            for column in df.columns:
                for i in range(start_index, min(start_index + trend_duration, len(df))):
                    df.loc[i, column] += trend_slope

    return df