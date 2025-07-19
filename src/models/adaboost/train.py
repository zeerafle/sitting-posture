import os
import sys
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from skopt.space import Integer

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, parent_dir)

from models.base_trainer import BaseTrainer


class AdaBoostTrainer(BaseTrainer):
    def get_estimator(self):
        return AdaBoostClassifier(random_state=self.params["random_state"])

    def get_param_space(self):
        return {
            "n_estimators": Integer(
                self.params["adaboost"]["n_estimators_min"],
                self.params["adaboost"]["n_estimators_max"],
            ),
        }

    def log_model_specific_metrics(self, model, live):
        live.log_metric("estimator_weights_mean", float(np.mean(model.estimator_weights_)), plot=False)
        live.log_metric("feature_importance_mean", float(np.mean(model.feature_importances_)), plot=False)

    def get_y_pred_proba(self, model, X_test):
        return model.decision_function(X_test)


if __name__ == "__main__":
    trainer = AdaBoostTrainer(model_name="adaboost")
    trainer.run()
