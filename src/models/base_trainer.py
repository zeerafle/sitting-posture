import os
import joblib
import dvc.api
from abc import ABC, abstractmethod
from loguru import logger
from .utils import configure_logging
from .training_workflow import run_standard, run_loso

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "..", '..'))

class BaseTrainer(ABC):
    def __init__(self, model_name: str, train_combined: bool=False, use_loso: bool=False):
        self.model_name     = model_name
        self.train_combined = train_combined
        self.use_loso       = use_loso
        # assume params_show returns a dict
        self.params         = dvc.api.params_show()
        self.dvclive_dir    = os.path.join(parent_dir, "dvclive", model_name)
        self.models_dir     = os.path.join(parent_dir, "models",  model_name)
        self.views          = ["front","left","right"]

        os.makedirs(self.models_dir, exist_ok=True)
        logger.info(f"Initialized '{model_name}' trainer "
                    f"(combined={train_combined}, loso={use_loso})")

    @abstractmethod
    def get_estimator(self):
        """Return a fresh sklearn‐style estimator"""
        pass

    @abstractmethod
    def get_param_space(self):
        """Return your BayesSearchCV space dict"""
        pass

    def save_model(self, model, fname: str):
        path = os.path.join(self.models_dir, fname)
        joblib.dump(model, path)
        logger.success(f"Saved model → {path}")

    def get_y_pred_proba(self, model, X):
        logger.debug(f"Getting prediction probabilities for {len(X)} test samples")
        return model.predict_proba(X)[:, 1]

    def run(self):
        """
        Dispatch either to the standard train/test flow
        or the LOSO flow (per‐subject folds).
        """
        if self.use_loso:
            run_loso(self)
        else:
            run_standard(self)
