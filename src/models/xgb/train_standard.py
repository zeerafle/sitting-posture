import argparse
import os
import sys

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, parent_dir)

import numpy as np
from xgboost import XGBClassifier
from skopt.space import Categorical, Real

from models.base_trainer import BaseTrainer
from models.standard_workflow import run_standard_workflow


class XGBoostStandardTrainer(BaseTrainer):
    def __init__(self):
        super().__init__(
            model_name="xgb",
            train_combined=True,  # Always use combined data
            use_loso=False        # This is for standard split
        )

    def get_estimator(self):
        return XGBClassifier(
            random_state=self.params["random_state"],
            eval_metric='logloss',
            objective='binary:logistic'
        )

    def get_param_space(self):
        return {
            "n_estimators": Categorical(self.params["xgb"]["n_estimators"]),
            "learning_rate": Real(
                self.params["xgb"]["learning_rate_min"],
                self.params["xgb"]["learning_rate_max"]
            ),
            "gamma": Real(
                self.params["xgb"]["gamma_min"],
                self.params["xgb"]["gamma_max"]
            ),
            "max_depth": Categorical(self.params["xgb"]["max_depths"]),
            "min_child_weight": Categorical(self.params["xgb"]["min_child_weights"]),
            "subsample": Categorical(self.params["xgb"]["subsamples"]),
            "lambda": Categorical(self.params["xgb"]["regulation_lambdas"]),
            "alpha": Categorical(self.params["xgb"]["regulation_alphas"]),
        }

    def log_model_specific_metrics(self, model, live):
        """Log XGBoost-specific metrics."""
        if hasattr(model, 'feature_importances_'):
            live.log_metric("feature_importance_mean", float(np.mean(model.feature_importances_)))
            live.log_metric("feature_importance_std", float(np.std(model.feature_importances_)))

    def run(self):
        """Run the standard workflow."""
        return run_standard_workflow(self)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost model with standard train-test split")
    args = parser.parse_args()

    trainer = XGBoostStandardTrainer()
    model, metrics = trainer.run()
