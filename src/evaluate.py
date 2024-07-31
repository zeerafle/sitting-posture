from dvclive import Live
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
)
import numpy as np

from models.utils import log_confusion_matrix, log_roc_auc_curve


def evaluate(
    model, y_test: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray, live: Live
):
    y_test = np.ravel(y_test)
    y_pred = np.ravel(y_pred)

    # log test metric
    live.log_metric("test_accuracy", accuracy_score(y_test, y_pred))
    live.log_metric("test_recall", recall_score(y_test, y_pred))
    live.log_metric("test_precision", precision_score(y_test, y_pred))
    live.log_metric("test_f1", f1_score(y_test, y_pred))
    live.log_metric("test_roc_auc", roc_auc_score(y_test, y_pred_proba))

    # Plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    log_confusion_matrix(live, cm, class_names=["ergonomic", "non-ergonomic"])
    log_roc_auc_curve(live, y_test, y_pred_proba)
