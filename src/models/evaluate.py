import os
from dvclive import Live
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate, GroupKFold
import numpy as np
import pandas as pd

from models.utils import log_confusion_matrix, log_roc_auc_curve


def evaluate(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train,
    y_test,
    y_pred,
    y_pred_proba,
    live: Live,
    groups_train=None,
    groups_test=None,
    cross_validate_enabled: bool = True,
    plot_cm: bool = True,
    plot_roc_auc: bool = True,
):
    """
    Evaluate model performance using various metrics with pandas.

    Parameters:
    -----------
    model: The trained model to evaluate
    X_train, X_test: Training and test features (pandas DataFrames)
    y_train, y_test: Training and test labels (array-like or pandas Series)
    y_pred: Model predictions on test set (array-like)
    y_pred_proba: Prediction probabilities on test set (array-like of positive class)
    live: DVCLive object for logging
    groups_train: Subject IDs for training data (for grouped CV)
    groups_test: Subject IDs for test data (for grouped CV)
    cross_validate_enabled: Whether to perform cross-validation
    plot_cm: Whether to plot confusion matrix
    plot_roc_auc: Whether to plot ROC curve
    """
    # Persist predictions for further analysis
    y_pred_save_path = os.path.join(live.dir, "y_pred.csv")
    pd.DataFrame(
        {
            "y_true": np.ravel(y_test),
            "y_pred": np.ravel(y_pred),
            "y_pred_proba": np.ravel(y_pred_proba),
        }
    ).to_csv(y_pred_save_path, index=False)
    live.log_artifact(y_pred_save_path, "predictions")

    # Ensure 1D arrays
    y_test = np.ravel(y_test)
    y_pred = np.ravel(y_pred)
    y_pred_proba = np.ravel(y_pred_proba)

    # Log test metrics
    live.log_metric("test/accuracy", accuracy_score(y_test, y_pred), plot=False)
    live.log_metric("test/recall", recall_score(y_test, y_pred), plot=False)
    live.log_metric("test/precision", precision_score(y_test, y_pred), plot=False)
    live.log_metric("test/f1", f1_score(y_test, y_pred), plot=False)
    live.log_metric("test/roc_auc", roc_auc_score(y_test, y_pred_proba), plot=False)

    # Visualizations
    if plot_cm:
        cm = confusion_matrix(y_test, y_pred)
        log_confusion_matrix(live, cm, class_names=["ergonomic", "non-ergonomic"])
    if plot_roc_auc:
        log_roc_auc_curve(live, y_test, y_pred_proba)

    if cross_validate_enabled:
        # Combine train and test data for cross-validation
        X_combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)
        y_combined = np.concatenate([np.ravel(y_train), np.ravel(y_test)])

        # Setup cross-validation strategy
        use_groups = groups_train is not None and groups_test is not None
        if use_groups:
            groups_combined = np.concatenate(
                [np.asarray(groups_train), np.asarray(groups_test)]
            )
            cv = GroupKFold(n_splits=5)
            cv_splits = list(cv.split(X_combined, y_combined, groups_combined))
            live.log_param("cv_strategy", "GroupKFold-by-subject")
        else:
            # Use StratifiedKFold instead of just an integer
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_splits = cv.split(X_combined, y_combined)
            live.log_param("cv_strategy", "StratifiedKFold")

        # Cross-validation scoring
        scoring = ["accuracy", "f1", "precision", "recall", "roc_auc"]

        try:
            cv_scores = cross_validate(
                model,
                X_combined,
                y_combined,
                cv=cv_splits,
                scoring=scoring,
                n_jobs=-1,
                error_score='raise'  # Raise errors to help debug issues
            )
        except Exception as e:
            # Log error and return None if cross-validation fails
            live.log_param("cv_error", str(e))
            return None

        live.log_metric("cv_accuracy_mean", np.mean(cv_scores["test_accuracy"]), plot=False)
        live.log_metric("cv_f1_mean", np.mean(cv_scores["test_f1"]), plot=False)
        live.log_metric("cv_precision_mean", np.mean(cv_scores["test_precision"]), plot=False)
        live.log_metric("cv_recall_mean", np.mean(cv_scores["test_recall"]), plot=False)
        # Log metrics if available
        if "test_roc_auc" in cv_scores:
            live.log_metric("cv_roc_auc_mean", np.mean(cv_scores["test_roc_auc"]), plot=False)
            live.log_metric("cv_std_roc_auc", np.std(cv_scores["test_roc_auc"]), plot=False)

        live.log_metric("cv_std_accuracy", np.std(cv_scores["test_accuracy"]), plot=False)
        live.log_metric("cv_std_f1", np.std(cv_scores["test_f1"]), plot=False)
        live.log_metric("cv_std_precision", np.std(cv_scores["test_precision"]), plot=False)
        live.log_metric("cv_std_recall", np.std(cv_scores["test_recall"]), plot=False)

        return cv_scores

    return None
