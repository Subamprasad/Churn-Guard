import logging
import pandas as pd
import numpy as np
from zenml import step

def generate_synthetic_data(n_samples=1000):
    """Generates synthetic churn data."""
    np.random.seed(42)
    data = {
        'age': np.random.normal(35, 12, n_samples).astype(int),
        'usage_minutes': np.random.normal(300, 100, n_samples),
        'contract_length': np.random.choice([1, 12, 24], n_samples),
        'support_calls': np.random.poisson(1, n_samples),
        'payment_delay': np.random.gamma(2, 2, n_samples),
        'churn': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]) # 20% churn rate
    }
    df = pd.DataFrame(data)
    # Add some correlation
    df.loc[df['support_calls'] > 3, 'churn'] = 1
    df.loc[df['usage_minutes'] < 50, 'churn'] = 1
    return df

class IngestData:
    """
    Data Ingestion class which ingests data from the source and returns a DataFrame.
    """
    def __init__(self, data_path: str = None):
        self.data_path = data_path

    def get_data(self) -> pd.DataFrame:
        if self.data_path:
            logging.info(f"Ingesting data from {self.data_path}")
            return pd.read_csv(self.data_path)
        else:
            logging.info("No data path provided. Generating synthetic data.")
            return generate_synthetic_data()

@step
def ingest_data(data_path: str = None) -> pd.DataFrame:
    """
    Args:
        data_path: path to the data
    Returns:
        df: pd.DataFrame
    """
    try:
        if data_path == "None": # Handle basic string 'None' from cli args if happens
            data_path = None
        ingest_data_obj = IngestData(data_path)
        df = ingest_data_obj.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        raise e
