import argparse
import os
import sys
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from skopt.space import Categorical

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, parent_dir)

from models.base_trainer import BaseTrainer
from models.loso_workflow import run_loso_workflow


class MLPTwoLayersWrapper(BaseEstimator, ClassifierMixin):
    """
    A wrapper class for MLPClassifier that handles two hidden layers.
    This wrapper converts simple scalar parameters to the tuple format
    required by MLPClassifier's hidden_layer_sizes parameter.
    """
    def __init__(self, layer1=100, layer2=50, learning_rate_init=0.001, alpha=0.0001):
        self.layer1 = layer1
        self.layer2 = layer2
        self.learning_rate_init = learning_rate_init
        self.alpha = alpha
        # Classes_ attribute will be set during fit
        self.classes_ = None
        self._create_model()

    def _create_model(self):
        """Create the MLPClassifier with current parameters"""
        self.model = MLPClassifier(
            hidden_layer_sizes=(self.layer1,), # only one hidden layer
            learning_rate_init=self.learning_rate_init,
            alpha=self.alpha,
            random_state=42,  # Will be overridden by set_params
            max_iter=1000,    # Will be overridden by set_params
            batch_size=32,    # Will be overridden by set_params
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )

    def fit(self, X, y):
        # Store classes for scikit-learn compatibility
        self.classes_ = np.unique(y)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def score(self, X, y):
        return self.model.score(X, y)


class NeuralNetworkLOSOTrainer(BaseTrainer):
    def __init__(self):
        super().__init__(
            model_name="nn",
            train_combined=True,  # Always use combined data
            use_loso=False        # This is for standard split
        )

    def get_estimator(self):
        wrapper = MLPTwoLayersWrapper()
        # Set the fixed parameters that aren't part of the optimization
        wrapper.model.set_params(
            random_state=self.params["random_state"],
            max_iter=self.params["nn"]["epochs"],
            batch_size=self.params["nn"]["batch_size"],
        )
        return wrapper

    def get_param_space(self):
        return {
            "layer1": Integer(
                self.params["nn"]["first_hidden_layer_sizes_min"],
                self.params["nn"]["first_hidden_layer_sizes_max"]
            ),
            "layer2": Integer(
                self.params["nn"]["second_hidden_layer_sizes_min"],
                self.params["nn"]["second_hidden_layer_sizes_max"]
            ),
            "learning_rate_init": Categorical(self.params["nn"]["learning_rates"]),
        }

    def log_model_specific_metrics(self, model, live):
        """Log neural network-specific metrics."""
        # Access the underlying MLPClassifier
        mlp_model = model.model
        if hasattr(mlp_model, 'loss_'):
            live.log_metric("final_loss", float(mlp_model.loss_))
        if hasattr(mlp_model, 'n_iter_'):
            live.log_metric("training_iterations", int(mlp_model.n_iter_))
        if hasattr(mlp_model, 'validation_scores_') and mlp_model.validation_scores_:
            live.log_metric("best_validation_score", float(max(mlp_model.validation_scores_)))

    def run(self):
        """Run the LOSO workflow."""
        return run_loso_workflow(self)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Neural Network model with LOSO grouped 5-fold cross validation")
    args = parser.parse_args()

    trainer = NeuralNetworkLOSOTrainer()
    model, metrics = trainer.run()
