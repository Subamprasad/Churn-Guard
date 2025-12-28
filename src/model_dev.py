from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from typing import Any

class Model(ABC):
    """
    Abstract Base Class for all models.
    """
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model.
        """
        pass

class RandomForestModel(Model):
    """
    RandomForestModel that implements the Model interface.
    """
    def train(self, X_train, y_train, **kwargs):
        clf = RandomForestClassifier(**kwargs)
        clf.fit(X_train, y_train)
        return clf

class XGBoostModel(Model):
    """
    XGBoostModel that implements the Model interface.
    """
    def train(self, X_train, y_train, **kwargs):
        clf = xgb.XGBClassifier(**kwargs)
        clf.fit(X_train, y_train)
        return clf

class LightGBMModel(Model):
    """
    LightGBMModel that implements the Model interface.
    """
    def train(self, X_train, y_train, **kwargs):
        clf = lgb.LGBMClassifier(**kwargs)
        clf.fit(X_train, y_train)
        return clf

class ModelFactory:
    """
    Factory class to create models.
    """
    @staticmethod
    def get_model(model_name: str) -> Model:
        if model_name == "lightgbm":
            return LightGBMModel()
        elif model_name == "xgboost":
            return XGBoostModel()
        elif model_name == "randomforest":
            return RandomForestModel()
        else:
            raise ValueError(f"Model {model_name} not supported.")
