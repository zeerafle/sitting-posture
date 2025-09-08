import os
import json
import argparse
import pandas as pd
from loguru import logger
import importlib

from .loso_workflow import run_loso_workflow


def load_best_model_config(config_path):
    """
    Load best model configuration from JSON file.

    Args:
        config_path: Path to best model configuration JSON

    Returns:
        dict: Best model configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Best model config not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config


def get_trainer_class(model_name):
    """
    Get the appropriate trainer class for the specified model.

    Args:
        model_name: Name of the model (e.g., 'xgb', 'nn', 'adaboost')

    Returns:
        class: Trainer class for the model
    """
    if model_name == 'xgb':
        from .xgb.train_loso import XGBoostLOSOTrainer
        return XGBoostLOSOTrainer
    elif model_name == 'nn':
        from .nn.train_loso import NeuralNetworkLOSOTrainer
        return NeuralNetworkLOSOTrainer
    elif model_name == 'adaboost':
        from .adaboost.train_loso import AdaBoostLOSOTrainer
        return AdaBoostLOSOTrainer
    else:
        raise ValueError(f"Unknown model name: {model_name}")


class AblationTrainer:
    """
    Trainer for ablation experiments using the best model from statistical analysis.
    """

    def __init__(self, best_model_config, data_mode, feature_mode):
        """
        Initialize ablation trainer.

        Args:
            best_model_config: Configuration of the best model
            data_mode: Data mode for ablation ('real', 'synthetic', 'all')
            feature_mode: Feature mode for ablation ('keypoints_only', 'all_features')
        """
        self.best_model_config = best_model_config
        self.data_mode = data_mode
        self.feature_mode = feature_mode
        self.model_name = best_model_config['model_name']

        # Get the trainer class for the best model
        trainer_class = get_trainer_class(self.model_name)
        self.trainer = trainer_class()

        # Update trainer settings for ablation
        self.trainer.model_name = self.model_name
        self.trainer.data_mode = data_mode
        self.feature_mode = feature_mode

        # Set up ablation-specific paths
        self.setup_ablation_paths()

        logger.info(f"Initialized ablation trainer for {self.model_name} with {data_mode}_{feature_mode}")

    def setup_ablation_paths(self):
        """Setup paths for ablation experiment."""
        # Update DVCLive directory for ablation
        base_dvclive_dir = os.path.join("dvclive", "ablation", f"{self.data_mode}_{self.feature_mode}")
        self.trainer.dvclive_dir = base_dvclive_dir

        # Update models directory for ablation
        base_models_dir = os.path.join("models", "ablation")
        self.trainer.models_dir = base_models_dir

        # Create directories
        os.makedirs(base_dvclive_dir, exist_ok=True)
        os.makedirs(base_models_dir, exist_ok=True)

    def get_data_path(self):
        """Get the data path for this ablation configuration."""
        return f"data/processed/ablation/{self.data_mode}_{self.feature_mode}/data.csv"

    def run_ablation(self):
        """
        Run the ablation experiment using LOSO with grouped 5-fold cross validation.

        Returns:
            tuple: (model, metrics)
        """
        logger.info(f"Starting ablation experiment: {self.data_mode}_{self.feature_mode}")
        logger.info(f"Using best model: {self.model_name}")

        # Get data path for this ablation configuration
        data_path = self.get_data_path()

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Ablation data not found: {data_path}. Please run prepare_ablation first.")

        # Log ablation configuration
        logger.info(f"Data mode: {self.data_mode}")
        logger.info(f"Feature mode: {self.feature_mode}")
        logger.info(f"Best model selection metric: {self.best_model_config['metric']}")
        logger.info(f"Best model selection reason: {self.best_model_config['selection_reason']}")

        # Run LOSO workflow with the specified data
        model, metrics = run_loso_workflow(self.trainer, data_path)

        # Save ablation-specific information
        self.save_ablation_info(metrics)

        logger.success(f"Completed ablation experiment: {self.data_mode}_{self.feature_mode}")

        return model, metrics

    def save_ablation_info(self, metrics):
        """
        Save ablation experiment information.

        Args:
            metrics: Metrics from the LOSO experiment
        """
        ablation_info = {
            'data_mode': self.data_mode,
            'feature_mode': self.feature_mode,
            'best_model_config': self.best_model_config,
            'experiment_results': metrics,
            'experiment_timestamp': pd.Timestamp.now().isoformat()
        }

        info_path = os.path.join(self.trainer.dvclive_dir, "ablation_info.json")
        with open(info_path, 'w') as f:
            json.dump(ablation_info, f, indent=2, default=str)

        logger.info(f"Saved ablation info to {info_path}")


def main():
    parser = argparse.ArgumentParser(description='Train ablation models using best model from statistical analysis')
    parser.add_argument('--data-mode', type=str, required=True,
                       choices=['real', 'synthetic', 'all'],
                       help='Data mode for ablation experiment')
    parser.add_argument('--feature-mode', type=str, required=True,
                       choices=['keypoints_only', 'all_features'],
                       help='Feature mode for ablation experiment')
    parser.add_argument('--best-model-config', type=str, required=True,
                       help='Path to best model configuration JSON file')

    args = parser.parse_args()

    try:
        # Load best model configuration
        logger.info(f"Loading best model configuration from {args.best_model_config}")
        best_model_config = load_best_model_config(args.best_model_config)

        logger.info(f"Best model: {best_model_config['model_name']}")
        logger.info(f"Selection metric: {best_model_config['metric']}")
        logger.info(f"Selection reason: {best_model_config['selection_reason']}")

        # Initialize ablation trainer
        ablation_trainer = AblationTrainer(
            best_model_config,
            args.data_mode,
            args.feature_mode
        )

        # Run ablation experiment
        model, metrics = ablation_trainer.run_ablation()

        # Print summary results
        if metrics and 'accuracy_mean' in metrics:
            print(f"\n=== ABLATION RESULTS SUMMARY ===")
            print(f"Configuration: {args.data_mode}_{args.feature_mode}")
            print(f"Model: {best_model_config['model_name']}")
            print(f"Mean Accuracy: {metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}")
            print(f"Mean F1: {metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}")
            print(f"Mean ROC-AUC: {metrics['roc_auc_mean']:.4f} ± {metrics['roc_auc_std']:.4f}")

        logger.success("Ablation training completed successfully")

    except Exception as e:
        logger.error(f"Error in ablation training: {e}")
        raise


if __name__ == "__main__":
    main()
