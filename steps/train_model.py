import logging
import pandas as pd
from zenml import step
from src.model_dev import ModelFactory
from sklearn.base import ClassifierMixin
from typing import Type

@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str = "xgboost",
) -> ClassifierMixin:
    """
    Args:
        X_train: pd.DataFrame
        X_test: pd.DataFrame
        y_train: pd.Series
        y_test: pd.Series
        model_name: str
    Returns:
        model: ClassifierMixin
    """
    try:
        model = ModelFactory.get_model(model_name)
        trained_model = model.train(X_train, y_train)
        return trained_model
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e
