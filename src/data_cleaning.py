import logging
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
    Abstract Class defining strategy for handling data
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreProcessStrategy(DataStrategy):
    """
    Strategy for preprocessing data
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data
        """
        try:
            # Example preprocessing: Drop columns, fill NA, encode
            # For simplicity in this synthetic example, we'll assume mostly numeric data
            # but we will handle categorical if present
            
            # Drop unnecessary cols if they exist
            if "customer_id" in data.columns:
                data = data.drop("customer_id", axis=1)
                
            # Fill missing values
            data = data.fillna(data.mean(numeric_only=True))
            
            # Simple encoding for object columns
            for col in data.select_dtypes(include=['object']).columns:
                data[col] = data[col].astype('category').cat.codes
                
            return data
        except Exception as e:
            logging.error(f"Error in preprocessing data: {e}")
            raise e

class DataDivideStrategy(DataStrategy):
    """
    Strategy for dividing data into train and test
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divide data into train and test
        """
        try:
            X = data.drop("churn", axis=1)
            y = data["churn"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in dividing data: {e}")
            raise e

class DataCleaning:
    """
    Class for executing data handling strategies
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle data based on strategy
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in handling data: {e}")
            raise e
