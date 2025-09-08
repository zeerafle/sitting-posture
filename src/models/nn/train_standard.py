import argparse
import os
import sys
import numpy as np
from sklearn.neural_network import MLPClassifier
from skopt.space import Categorical, Real

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, parent_dir)

from models.base_trainer import BaseTrainer
from models.standard_workflow import run_standard_workflow


class NeuralNetworkStandardTrainer(BaseTrainer):
    def __init__(self):
        super().__init__(
            model_name="nn",
            train_combined=True,  # Always use combined data
            use_loso=False        # This is for standard split
        )

    def get_estimator(self):
        return MLPClassifier(
            random_state=self.params["random_state"],
            max_iter=self.params["nn"]["epochs"],
            batch_size=self.params["nn"]["batch_size"],
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )

    def get_param_space(self):
        # Create combinations of hidden layer sizes
        first_layer_sizes = list(range(
            self.params["nn"]["first_hidden_layer_sizes_min"],
            self.params["nn"]["first_hidden_layer_sizes_max"] + 1,
            64
        ))
        second_layer_sizes = list(range(
            self.params["nn"]["second_hidden_layer_sizes_min"],
            self.params["nn"]["second_hidden_layer_sizes_max"] + 1,
            128
        ))

        # Create tuples for hidden layer configurations
        hidden_layer_configs = []
        for first in first_layer_sizes:
            for second in second_layer_sizes:
                hidden_layer_configs.append((first, second))

        return {
            "hidden_layer_sizes": Categorical(hidden_layer_configs),
            "learning_rate_init": Categorical(self.params["nn"]["learning_rates"]),
            "alpha": Real(1e-6, 1e-2, prior='log-uniform')  # L2 regularization
        }

    def log_model_specific_metrics(self, model, live):
        """Log neural network-specific metrics."""
        if hasattr(model, 'loss_'):
            live.log_metric("final_loss", float(model.loss_))
        if hasattr(model, 'n_iter_'):
            live.log_metric("training_iterations", int(model.n_iter_))
        if hasattr(model, 'validation_scores_') and model.validation_scores_:
            live.log_metric("best_validation_score", float(max(model.validation_scores_)))

    def run(self):
        """Run the standard workflow."""
        return run_standard_workflow(self)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Neural Network model with standard train-test split")
    args = parser.parse_args()

    trainer = NeuralNetworkStandardTrainer()
    model, metrics = trainer.run()
