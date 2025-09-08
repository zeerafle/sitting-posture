import os
import json
import time
import pandas as pd
from codecarbon import OfflineEmissionsTracker
from dvclive import Live
from loguru import logger
from sklearn.model_selection import StratifiedGroupKFold

from .evaluate import evaluate
from .utils import bayes_search, NumpyEncoder


def run_standard_workflow(trainer):
    """
    Runs the standard train/test split workflow with hyperparameter tuning.

    Args:
        trainer: BaseTrainer instance
    """
    logger.info("Starting standard workflow with train-test split")

    # Load preprocessed train and test data
    train_path = "data/processed/standard_split/train.csv"
    test_path = "data/processed/standard_split/test.csv"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Preprocessed data not found. Please run prepare_standard_split first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    logger.info(f"Loaded train data: {train_df.shape}, test data: {test_df.shape}")

    # Separate features and labels
    feature_cols = [col for col in train_df.columns if col != 'labels']
    X_train = train_df[feature_cols].values
    y_train = train_df['labels'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['labels'].values

    logger.info(f"Features: {len(feature_cols)}, Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Setup DVCLive logging
    dvclive_path = os.path.join(trainer.dvclive_dir, "standard")
    os.makedirs(dvclive_path, exist_ok=True)

    with Live(dvclive_path) as live:
        # Phase 1: Hyperparameter tuning with cross-validation
        logger.info("Phase 1: Hyperparameter tuning with cross-validation")

        # For hyperparameter tuning, we need to create groups for CV
        # Since we don't have subject IDs in the processed data, we'll use stratified CV
        best_params, cv_results = bayes_search(
            trainer.get_estimator(),
            trainer.get_param_space(),
            X_train, y_train,
            trainer.params["n_iter"],
            trainer.params["cv"],
            trainer.params["scoring"],
            live
        )

        # Save hyperparameter tuning results
        hyperparameter_results = {
            'best_params': best_params,
            'cv_results': cv_results,
            'timestamp': time.time()
        }

        hyperparameter_path = os.path.join(dvclive_path, "hyperparameter_results.json")
        with open(hyperparameter_path, "w") as f:
            json.dump(hyperparameter_results, f, indent=2, cls=NumpyEncoder)

        logger.info(f"Best hyperparameters: {best_params}")

        # Phase 2: Train final model with best parameters
        logger.info("Phase 2: Training final model with best parameters")

        final_model = trainer.get_estimator()
        final_model.set_params(**best_params)

        # Track training emissions
        with OfflineEmissionsTracker(save_to_file=False) as train_tracker:
            start_time = time.time()
            final_model.fit(X_train, y_train)
            training_time = time.time() - start_time

        live.log_metric("train/time_seconds", training_time)
        live.log_metric("train/emissions_kwh", train_tracker.final_emissions_data.energy_consumed)

        # Phase 3: Model evaluation on test set
        logger.info("Phase 3: Model evaluation on test set")

        # Track inference emissions
        with OfflineEmissionsTracker(save_to_file=False) as inference_tracker:
            start_time = time.time()
            y_pred = final_model.predict(X_test)
            y_proba = final_model.predict_proba(X_test)[:, 1]
            inference_time = time.time() - start_time

        live.log_metric("test/time_seconds", inference_time)
        live.log_metric("test/emissions_kwh", inference_tracker.final_emissions_data.energy_consumed)

        # Evaluate model performance
        metrics = evaluate(
            final_model,
            X_train, X_test,
            y_train, y_test,
            y_pred, y_proba,
            live
        )

        # Save test predictions
        predictions_df = pd.DataFrame({
            'y_true': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba
        })
        predictions_path = os.path.join(dvclive_path, "test_predictions.csv")
        predictions_df.to_csv(predictions_path, index=False)

        # Log model-specific metrics if available
        if hasattr(trainer, 'log_model_specific_metrics'):
            trainer.log_model_specific_metrics(final_model, live)

        logger.info(f"Model evaluation completed. Test accuracy: {metrics.get('test_accuracy', 'N/A'):.4f}")

    # Save final trained model
    model_filename = f"{trainer.model_name}_standard.joblib"
    trainer.save_model(final_model, model_filename)

    logger.success("Standard workflow completed successfully")
    return final_model, metrics


def load_standard_data():
    """
    Load preprocessed standard split data.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_columns)
    """
    train_path = "data/processed/standard_split/train.csv"
    test_path = "data/processed/standard_split/test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_cols = [col for col in train_df.columns if col != 'labels']

    X_train = train_df[feature_cols].values
    y_train = train_df['labels'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['labels'].values

    return X_train, X_test, y_train, y_test, feature_cols
