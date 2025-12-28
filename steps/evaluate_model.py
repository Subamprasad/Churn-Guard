import logging
import pandas as pd
from zenml import step
from src.evaluation import Recall, F1
from sklearn.base import ClassifierMixin
from typing import Tuple
from typing_extensions import Annotated

@step
def evaluate_model(
    model: ClassifierMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[
    Annotated[float, "recall"],
    Annotated[float, "f1_score"],
]:
    """
    Args:
        model: ClassifierMixin
        X_test: pd.DataFrame
        y_test: pd.Series
    Returns:
        recall: float
        f1_score: float
    """
    try:
        prediction = model.predict(X_test)
        
        recall_class = Recall()
        recall = recall_class.calculate_scores(y_test, prediction)
        
        f1_class = F1()
        f1 = f1_class.calculate_scores(y_test, prediction)
        
        return recall, f1
    except Exception as e:
        logging.error(f"Error in evaluating model: {e}")
        raise e
