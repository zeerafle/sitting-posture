import os
import joblib
import dvc.api
from abc import ABC, abstractmethod
from loguru import logger

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "..", '..'))

class BaseTrainer(ABC):
    def __init__(self, model_name: str, train_combined: bool=True, use_loso: bool=False, data_path_suffix: str=""):
        self.model_name     = model_name
        self.train_combined = train_combined  # Always True in new structure (combined data only)
        self.use_loso       = use_loso
        # assume params_show returns a dict
        self.params         = dvc.api.params_show(os.path.join(parent_dir, "params.yaml"))
        self.dvclive_dir    = os.path.join(parent_dir, "dvclive", model_name)
        self.models_dir     = os.path.join(parent_dir, "models",  model_name)
        self.data_path_suffix = data_path_suffix  # For ablation studies (e.g., "real_keypoints_only")

        os.makedirs(self.models_dir, exist_ok=True)
        logger.info(f"Initialized '{model_name}' trainer "
            f"(combined={train_combined}, loso={use_loso}, data_path_suffix={data_path_suffix})")

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
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)[:, 1]
        else:
            # Fallback for models without predict_proba
            logger.warning(f"Model {self.model_name} doesn't have predict_proba, using decision_function or predict")
            if hasattr(model, 'decision_function'):
                return model.decision_function(X)
            else:
                return model.predict(X)

    def run(self):
        """
        Abstract method to be implemented by subclasses.
        Should call appropriate workflow (standard or LOSO).
        """
        raise NotImplementedError("Subclasses must implement the run method")
