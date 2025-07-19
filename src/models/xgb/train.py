import os
import sys
import numpy as np
from xgboost import XGBClassifier
from skopt.space import Categorical, Real

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, parent_dir)

from models.base_trainer import BaseTrainer


class XGBoostTrainer(BaseTrainer):
    def get_estimator(self):
        return XGBClassifier(random_state=self.params["random_state"])

    def get_param_space(self):
        return {
            "n_estimators": Categorical(self.params["xgb"]["n_estimators"]),
            "learning_rate": Real(
                self.params["xgb"]["learning_rate_min"], self.params["xgb"]["learning_rate_max"]
            ),
            "gamma": Real(self.params["xgb"]["gamma_min"], self.params["xgb"]["gamma_max"]),
            "max_depth": Categorical(self.params["xgb"]["max_depths"]),
            "min_child_weight": Categorical(self.params["xgb"]["min_child_weights"]),
            "subsample": Categorical(self.params["xgb"]["subsamples"]),
            "lambda": Categorical(self.params["xgb"]["regulation_lambdas"]),
            "alpha": Categorical(self.params["xgb"]["regulation_alphas"]),
        }

    def save_model(self, model, view):
        model_path = os.path.join(self.models_dir, f"{self.model_name}_{view}.json")
        model.save_model(model_path)

    def log_model_specific_metrics(self, model, live):
        live.log_metric("feature_importance_mean", float(np.mean(model.feature_importances_)), plot=False)


if __name__ == "__main__":
    trainer = XGBoostTrainer(model_name="xgb")
    trainer.run()
