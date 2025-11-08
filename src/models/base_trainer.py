"""
Base training module with common functionality for all models.
Performs cross-validation on all available data.
"""
import os
import json
import numpy as np
from typing import Dict, Any, Optional
from dvclive.live import Live
from codecarbon import OfflineEmissionsTracker
import joblib

from models.utils import NumpyEncoder


class BaseTrainer:
    """Base class for training ML models with cross-validation."""

    def __init__(self, model, model_name: str, dvclive_path: str, models_dir: str):
        """
        Initialize the trainer.

        Args:
            model: The scikit-learn compatible model instance
            model_name: Name of the model (e.g., 'adaboost', 'xgb', 'nn')
            dvclive_path: Path to save DVCLive artifacts
            models_dir: Path to save trained models
        """
        self.model = model
        self.model_name = model_name
        self.dvclive_path = dvclive_path
        self.models_dir = models_dir

    def train(self, X, y) -> Any:
        """
        Train the model on all data.

        Args:
            X: Features
            y: Labels

        Returns:
            Trained model
        """
        emissions_path = os.path.join(self.dvclive_path, "emissions.csv")
        with OfflineEmissionsTracker(output_file=emissions_path) as tracker:
            self.model.fit(X, np.ravel(y))
        return self.model

    def predict(self, X):
        """
        Make predictions with emission tracking.

        Args:
            X: Features to predict on

        Returns:
            Tuple of (predictions, prediction probabilities)
        """
        emissions_path = os.path.join(self.dvclive_path, "emissions_inference.csv")

        # Get prediction probabilities first
        y_pred_proba = self.model.predict_proba(X)[:, 1]

        # Make predictions with emission tracking
        with OfflineEmissionsTracker(output_file=emissions_path) as tracker:
            y_pred = self.model.predict(X)

        return y_pred, y_pred_proba

    def save_model(self, model_path: str):
        """
        Save the trained model.

        Args:
            model_path: Path where to save the model
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Save based on file extension
        if model_path.endswith('.joblib'):
            with open(model_path, 'wb') as f:
                joblib.dump(self.model, f)
        elif model_path.endswith('.json'):
            self.model.save_model(model_path)
        else:
            raise ValueError(f"Unsupported model format: {model_path}")

    def log_model_specific_metrics(self, live: Live):
        """
        Log model-specific metrics. Override in subclasses.

        Args:
            live: DVCLive instance
        """
        pass

    def log_predictions(self, y_pred: np.ndarray, y_pred_proba: np.ndarray):
        """
        Save predictions to CSV files.

        Args:
            y_pred: Predictions
            y_pred_proba: Prediction probabilities
        """
        np.savetxt(
            os.path.join(self.dvclive_path, "y_pred.csv"),
            y_pred,
            delimiter=",",
        )
        np.savetxt(
            os.path.join(self.dvclive_path, "y_pred_proba.csv"),
            y_pred_proba,
            delimiter=",",
        )

    def log_cv_results(self, cv_scores: Dict, live: Live):
        """
        Log cross-validation results.

        Args:
            cv_scores: Dictionary of cross-validation scores
            live: DVCLive instance
        """
        cv_results_path = os.path.join(self.dvclive_path, "cv_results.json")
        with open(cv_results_path, 'w') as f:
            json.dump(cv_scores, f, indent=4, cls=NumpyEncoder)
        live.log_artifact(cv_results_path, type="cv_results")

    def log_emissions(self, live: Live):
        """
        Log emission artifacts.

        Args:
            live: DVCLive instance
        """
        emissions_path = os.path.join(self.dvclive_path, "emissions.csv")
        emissions_inference_path = os.path.join(self.dvclive_path, "emissions_inference.csv")

        if os.path.exists(emissions_path):
            live.log_artifact(emissions_path, type="emissions")
        if os.path.exists(emissions_inference_path):
            live.log_artifact(emissions_inference_path, type="emissions_inference")

    def run_training_pipeline(self, X, y, evaluate_fn, live: Live,
                              model_path: str, holdout_size: Optional[float] = 0.2):
        """
        Run the complete training pipeline with cross-validation.

        Args:
            X: All features
            y: All labels
            evaluate_fn: Evaluation function that performs CV
            live: DVCLive instance
            model_path: Path to save the trained model
            holdout_size: Optional holdout set size for additional evaluation (0 to disable)
        """
        # Train on all data
        print(f"Training {self.model_name} on all data...")
        self.train(X, y)

        # Log model-specific metrics
        self.log_model_specific_metrics(live)

        # Save the trained model
        print(f"Saving model to {model_path}...")
        self.save_model(model_path)

        # Log the saved model as an artifact
        live.log_artifact(model_path, type="model", name=self.model_name)

        # Generate predictions on all data for DVC compatibility
        # This creates the expected output files even though we use CV for evaluation
        print(f"Generating predictions...")
        y_pred, y_pred_proba = self.predict(X)
        self.log_predictions(y_pred, y_pred_proba)

        # Perform cross-validation evaluation
        print(f"Performing cross-validation...")
        cv_scores = evaluate_fn(self.model, X, y, live)

        # Log results
        self.log_emissions(live)
        self.log_cv_results(cv_scores, live)

        print(f"Training completed successfully!")
        return cv_scores


class AdaBoostTrainer(BaseTrainer):
    """Trainer for AdaBoost models."""

    def log_model_specific_metrics(self, live: Live):
        """Log AdaBoost-specific metrics."""
        if hasattr(self.model, 'estimator_weights_'):
            live.log_metric(
                "estimator_weights_mean",
                float(np.mean(self.model.estimator_weights_))
            )
        if hasattr(self.model, 'feature_importances_'):
            live.log_metric(
                "feature_importance_mean",
                float(np.mean(self.model.feature_importances_))
            )


class XGBTrainer(BaseTrainer):
    """Trainer for XGBoost models."""

    def log_model_specific_metrics(self, live: Live):
        """Log XGBoost-specific metrics."""
        if hasattr(self.model, 'feature_importances_'):
            live.log_metric(
                "feature_importance_mean",
                float(np.mean(self.model.feature_importances_))
            )


class NNTrainer(BaseTrainer):
    """Trainer for Neural Network models."""

    def log_model_specific_metrics(self, live: Live):
        """Log Neural Network-specific metrics."""
        if hasattr(self.model, 'best_loss_') and self.model.best_loss_ is not None:
            live.log_metric("best_loss", float(self.model.best_loss_))
        if hasattr(self.model, 'loss_curve_') and self.model.loss_curve_ is not None:
            for loss in self.model.loss_curve_:
                live.log_metric("loss_curve", float(loss))
