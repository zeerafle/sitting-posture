import os
import sys
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from skopt.space import Integer, Categorical
import dvc.api

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, parent_dir)

from models.base_trainer import BaseTrainer


class MLPWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, layer1=10, learning_rate_init=0.001):
        params = dvc.api.params_show()
        self.layer1 = layer1
        self.learning_rate_init = learning_rate_init
        self.model = MLPClassifier(
            hidden_layer_sizes=[self.layer1],
            batch_size=params["nn"]["batch_size"],
            max_iter=params["nn"]["epochs"],
            random_state=params["random_state"],
            learning_rate_init=self.learning_rate_init,
        )
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)


class NNTrainer(BaseTrainer):
    def get_estimator(self):
        return MLPWrapper()

    def get_param_space(self):
        return {
            "layer1": Integer(
                self.params["nn"]["first_hidden_layer_sizes_min"],
                self.params["nn"]["first_hidden_layer_sizes_max"],
            ),
            "learning_rate_init": Categorical(self.params["nn"]["learning_rates"]),
        }

    def log_model_specific_metrics(self, model, live):
        # The underlying MLPClassifier is in model.model
        if hasattr(model, "best_loss_") and model.best_loss_ is not None:
            live.log_metric("best_loss", float(model.best_loss_))
        if hasattr(model, "loss_curve_") and model.loss_curve_ is not None:
            for loss in model.loss_curve_:
                live.log_metric("loss_curve", float(loss))

    def set_best_params(self, model, best_params):
        model = model.model
        model.set_params(hidden_layer_sizes=best_params["layer1"],
                         learning_rate_init=best_params["learning_rate_init"])
        return model

if __name__ == "__main__":
    trainer = NNTrainer(model_name="nn")
    trainer.run()
