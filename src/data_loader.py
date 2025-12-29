
import pandas as pd
import os

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads the dataset from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} was not found.")
    
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded data from {filepath}. Shape: {df.shape}")
        return df
    except Exception as e:
        raise Exception(f"Error loading data: {e}")
