import os
import json
import time
import pandas as pd
import numpy as np
from loguru import logger
from dvclive import Live
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from codecarbon import OfflineEmissionsTracker

from utils import NumpyEncoder


def run_loso_workflow(trainer, data_path=None):
    """
    Runs Leave-One-Subject-Out workflow using grouped 5-fold cross validation.
    Each fold holds out disjoint participants to prevent subject-level leakage.

    Args:
        trainer: BaseTrainer instance
        data_path: Optional path to data file. If None, uses default LOSO data path.
    """
    logger.info("Starting LOSO workflow with grouped 5-fold cross validation")

    # Determine data path
    if data_path is None:
        data_path = "data/processed/loso/data.csv"

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"LOSO data not found at {data_path}. Please run prepare_loso first.")

    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded LOSO dataset: {df.shape}")

    # Separate features, labels, and groups
    feature_cols = [col for col in df.columns if col not in ['subject_id', 'labels', 'class_name', 'file_name', 'view_type']]
    X = df[feature_cols].values
    y = df['labels'].values
    groups = df['subject_id'].values

    logger.info(f"Features: {len(feature_cols)}, Samples: {len(X)}, Unique subjects: {len(np.unique(groups))}")

    # Verify class and subject distribution
    _verify_data_distribution(df, groups, y)

    # Setup DVCLive logging
    dvclive_path = os.path.join(trainer.dvclive_dir, "loso")
    os.makedirs(dvclive_path, exist_ok=True)

    # Setup grouped 5-fold cross validation
    cv_folds = trainer.params.get("cv_folds", 5)
    skf = StratifiedGroupKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=trainer.params.get("random_state", 42)
    )

    # Get subject-level stratification for proper folding
    subject_labels = df.groupby('subject_id')['labels'].first()
    unique_subjects = subject_labels.index.values
    subject_label_values = subject_labels.values

    try:
        # Generate fold splits based on subjects
        subject_splits = list(skf.split(unique_subjects, subject_label_values, unique_subjects))
        logger.info(f"Successfully created {len(subject_splits)} folds for LOSO")
    except ValueError as e:
        logger.error(f"Error creating grouped folds: {e}")
        # Fallback to simpler approach if grouping fails
        logger.warning("Falling back to subject-based manual splitting")
        subject_splits = _create_manual_subject_splits(unique_subjects, subject_label_values, cv_folds)

    # Store fold-wise metrics
    fold_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': [],
        'fold_info': []
    }

    with Live(dvclive_path) as live:
        # Run cross-validation folds
        for fold_idx, (train_subject_idx, test_subject_idx) in enumerate(subject_splits, 1):
            logger.info(f"Processing fold {fold_idx}/{cv_folds}")

            # Get subjects for this fold
            train_subjects = unique_subjects[train_subject_idx]
            test_subjects = unique_subjects[test_subject_idx]

            # Verify no subject leakage
            subject_intersection = set(train_subjects) & set(test_subjects)
            if subject_intersection:
                logger.error(f"SUBJECT LEAKAGE DETECTED in fold {fold_idx}! Subjects {subject_intersection} appear in both sets!")
                raise ValueError(f"Subject leakage detected in fold {fold_idx}")

            logger.info(f"Fold {fold_idx}: {len(train_subjects)} train subjects, {len(test_subjects)} test subjects")

            # Get sample indices for this fold
            train_mask = df['subject_id'].isin(train_subjects)
            test_mask = df['subject_id'].isin(test_subjects)

            X_train_fold = X[train_mask]
            y_train_fold = y[train_mask]
            X_test_fold = X[test_mask]
            y_test_fold = y[test_mask]

            # Log fold information
            fold_info = {
                'fold': fold_idx,
                'train_subjects': train_subjects.tolist(),
                'test_subjects': test_subjects.tolist(),
                'train_samples': len(X_train_fold),
                'test_samples': len(X_test_fold),
                'train_class_distribution': dict(pd.Series(y_train_fold).value_counts()),
                'test_class_distribution': dict(pd.Series(y_test_fold).value_counts())
            }
            fold_metrics['fold_info'].append(fold_info)

            logger.info(f"Fold {fold_idx} - Train: {len(X_train_fold)} samples, Test: {len(X_test_fold)} samples")

            # Train model for this fold
            model = trainer.get_estimator()

            # Set hyperparameters if available from previous tuning
            best_params = _load_best_params(trainer)
            if best_params:
                model.set_params(**best_params)

            # Track training time and emissions
            with OfflineEmissionsTracker(save_to_file=False) as train_tracker:
                start_time = time.time()
                model.fit(X_train_fold, y_train_fold)
                training_time = time.time() - start_time

            # Track inference time and emissions
            with OfflineEmissionsTracker(save_to_file=False) as inference_tracker:
                start_time = time.time()
                y_pred_fold = model.predict(X_test_fold)
                y_proba_fold = model.predict_proba(X_test_fold)[:, 1] if hasattr(model, 'predict_proba') else None
                inference_time = time.time() - start_time

            # Calculate metrics for this fold
            fold_accuracy = accuracy_score(y_test_fold, y_pred_fold)
            fold_precision = precision_score(y_test_fold, y_pred_fold, average='binary')
            fold_recall = recall_score(y_test_fold, y_pred_fold, average='binary')
            fold_f1 = f1_score(y_test_fold, y_pred_fold, average='binary')
            fold_roc_auc = roc_auc_score(y_test_fold, y_proba_fold) if y_proba_fold is not None else 0.0

            # Store metrics
            fold_metrics['accuracy'].append(fold_accuracy)
            fold_metrics['precision'].append(fold_precision)
            fold_metrics['recall'].append(fold_recall)
            fold_metrics['f1'].append(fold_f1)
            fold_metrics['roc_auc'].append(fold_roc_auc)

            # Log fold metrics to DVCLive
            live.log_metric(f"fold_{fold_idx}/accuracy", fold_accuracy)
            live.log_metric(f"fold_{fold_idx}/precision", fold_precision)
            live.log_metric(f"fold_{fold_idx}/recall", fold_recall)
            live.log_metric(f"fold_{fold_idx}/f1", fold_f1)
            live.log_metric(f"fold_{fold_idx}/roc_auc", fold_roc_auc)
            live.log_metric(f"fold_{fold_idx}/train_time", training_time)
            live.log_metric(f"fold_{fold_idx}/inference_time", inference_time)
            live.log_metric(f"fold_{fold_idx}/train_emissions", train_tracker.final_emissions_data.energy_consumed)
            live.log_metric(f"fold_{fold_idx}/inference_emissions", inference_tracker.final_emissions_data.energy_consumed)

            logger.info(f"Fold {fold_idx} results - Acc: {fold_accuracy:.4f}, F1: {fold_f1:.4f}, ROC-AUC: {fold_roc_auc:.4f}")

        # Calculate aggregated metrics
        aggregated_metrics = {}
        for metric_name, values in fold_metrics.items():
            if metric_name != 'fold_info' and values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                aggregated_metrics[f"{metric_name}_mean"] = mean_val
                aggregated_metrics[f"{metric_name}_std"] = std_val

                # Log to DVCLive
                live.log_metric(f"loso_mean_{metric_name}", mean_val)
                live.log_metric(f"loso_std_{metric_name}", std_val)

        logger.info(f"LOSO Results - Mean Accuracy: {aggregated_metrics['accuracy_mean']:.4f} ± {aggregated_metrics['accuracy_std']:.4f}")
        logger.info(f"LOSO Results - Mean F1: {aggregated_metrics['f1_mean']:.4f} ± {aggregated_metrics['f1_std']:.4f}")

        # Save detailed fold metrics
        fold_metrics_path = os.path.join(dvclive_path, "fold_metrics.json")
        with open(fold_metrics_path, "w") as f:
            json.dump(fold_metrics, f, indent=2, cls=NumpyEncoder)

        # Save aggregated metrics
        aggregated_metrics_path = os.path.join(dvclive_path, "aggregated_metrics.json")
        with open(aggregated_metrics_path, "w") as f:
            json.dump(aggregated_metrics, f, indent=2, cls=NumpyEncoder)

    # Train final model on all data
    logger.info("Training final model on complete dataset")
    final_model = trainer.get_estimator()
    if best_params:
        final_model.set_params(**best_params)

    final_model.fit(X, y)

    # Save final model
    model_filename = f"{trainer.model_name}_loso.joblib"
    trainer.save_model(final_model, model_filename)

    logger.success("LOSO workflow completed successfully")
    return final_model, aggregated_metrics


def _verify_data_distribution(df, groups, y):
    """Verify data distribution for LOSO analysis."""
    # Check subject-class consistency
    subject_class_consistency = df.groupby('subject_id')['labels'].nunique()
    inconsistent_subjects = subject_class_consistency[subject_class_consistency > 1]

    if len(inconsistent_subjects) > 0:
        logger.warning(f"Found {len(inconsistent_subjects)} subjects with inconsistent class labels")

    # Check class distribution
    unique_subjects = np.unique(groups)
    subject_labels = df.groupby('subject_id')['labels'].first()
    class_counts = subject_labels.value_counts()

    logger.info(f"Subject distribution by class: {dict(class_counts)}")

    # Check minimum subjects per class for CV
    min_subjects_per_class = class_counts.min()
    if min_subjects_per_class < 5:
        logger.warning(f"Minimum subjects per class ({min_subjects_per_class}) may be insufficient for 5-fold CV")


def _load_best_params(trainer):
    """Load best hyperparameters from previous standard training if available."""
    # Check if this is an ablation run by examining the dvclive_dir path
    is_ablation = "ablation" in trainer.dvclive_dir

    if is_ablation:
        # For ablation, get the model name to load the corresponding standard model hyperparameters
        model_name = trainer.model_name
        # Use the standard path instead of the ablation path
        params_path = os.path.join("dvclive", f"{model_name}", "standard", "hyperparameter_results.json")
    else:
        # For standard training, use the normal path
        params_path = os.path.join(trainer.dvclive_dir, "hyperparameter_results.json")

    logger.debug(f"Loading best hyperparameters from {params_path}")
    if os.path.exists(params_path):
        try:
            with open(params_path, "r") as f:
                results = json.load(f)
            best_params = results.get("best_params", {})
            logger.info(f"Loaded best hyperparameters from {'ablation' if is_ablation else 'standard'} training: {best_params}")
            return best_params
        except Exception as e:
            logger.warning(f"Could not load best parameters: {e}")

    logger.info("No previous hyperparameters found, using model defaults")
    return {}


def _create_manual_subject_splits(unique_subjects, subject_labels, n_splits):
    """Create manual subject splits when StratifiedGroupKFold fails."""
    from sklearn.model_selection import StratifiedKFold

    logger.info("Creating manual subject-based splits")

    # Use simple stratified split on subjects
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(skf.split(unique_subjects, subject_labels))

    logger.info(f"Created {len(splits)} manual splits")
    return splits


def load_loso_data(data_path=None):
    """
    Load preprocessed LOSO data.

    Args:
        data_path: Optional path to data file

    Returns:
        tuple: (X, y, groups, feature_columns)
    """
    if data_path is None:
        data_path = "data/processed/loso/data.csv"

    df = pd.read_csv(data_path)

    feature_cols = [col for col in df.columns if col not in ['subject_id', 'labels', 'class_name', 'file_name', 'view_type']]
    X = df[feature_cols].values
    y = df['labels'].values
    groups = df['subject_id'].values

    return X, y, groups, feature_cols
