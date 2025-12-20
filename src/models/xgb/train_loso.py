import argparse
import os
import sys
import numpy as np
from xgboost import XGBClassifier
from skopt.space import Categorical, Real

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, parent_dir)

from models.base_trainer import BaseTrainer
from models.loso_workflow import run_loso_workflow


class XGBoostLOSOTrainer(BaseTrainer):
    def __init__(self):
        super().__init__(
            model_name="xgb",
            train_combined=True,  # Always use combined data
            use_loso=True         # This is for LOSO
        )

    def get_estimator(self):
        return XGBClassifier(
            random_state=self.params["random_state"],
            eval_metric='logloss'
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
        """Run the LOSO workflow."""
        return run_loso_workflow(self)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost model with LOSO grouped 5-fold cross validation")
    args = parser.parse_args()

    trainer = XGBoostLOSOTrainer()
    model, metrics = trainer.run()
