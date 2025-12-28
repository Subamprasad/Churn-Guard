import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, recall_score, f1_score

class Evaluation(ABC):
    """
    Abstract Class defining strategy for evaluation our models
    """
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the scores for the model
        """
        pass

class MSE(Evaluation):
    """
    Evaluation Strategy that uses Mean Squared Error
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error in calculating MSE: {e}")
            raise e

class Recall(Evaluation):
    """
    Evaluation Strategy that uses Recall (Critical for Churn)
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating Recall")
            recall = recall_score(y_true, y_pred)
            logging.info(f"Recall: {recall}")
            return recall
        except Exception as e:
            logging.error(f"Error in calculating Recall: {e}")
            raise e

class F1(Evaluation):
    """
    Evaluation Strategy that uses F1 Score
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating F1 Score")
            f1 = f1_score(y_true, y_pred)
            logging.info(f"F1 Score: {f1}")
            return f1
        except Exception as e:
            logging.error(f"Error in calculating F1 Score: {e}")
            raise e
