import argparse
import os
import sys
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from skopt.space import Integer, Real

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, parent_dir)

from models.base_trainer import BaseTrainer
from models.loso_workflow import run_loso_workflow


class AdaBoostLOSOTrainer(BaseTrainer):
    def __init__(self):
        super().__init__(
            model_name="adaboost",
            train_combined=True,  # Always use combined data
            use_loso=True         # This is for LOSO
        )

    def get_estimator(self):
        return AdaBoostClassifier(
            random_state=self.params["random_state"],
        )

    def get_param_space(self):
        return {
            "n_estimators": Integer(
                self.params["adaboost"]["n_estimators_min"],
                self.params["adaboost"]["n_estimators_max"],
            )
        }

    def log_model_specific_metrics(self, model, live):
        """Log AdaBoost-specific metrics."""
        if hasattr(model, 'feature_importances_'):
            live.log_metric("feature_importance_mean", float(np.mean(model.feature_importances_)))
            live.log_metric("feature_importance_std", float(np.std(model.feature_importances_)))
        if hasattr(model, 'estimator_weights_'):
            live.log_metric("estimator_weights_mean", float(np.mean(model.estimator_weights_)))
            live.log_metric("estimator_weights_std", float(np.std(model.estimator_weights_)))

    def run(self):
        """Run the LOSO workflow."""
        return run_loso_workflow(self)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AdaBoost model with LOSO grouped 5-fold cross validation")
    args = parser.parse_args()

    trainer = AdaBoostLOSOTrainer()
    model, metrics = trainer.run()
