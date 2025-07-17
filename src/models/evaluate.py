from dvclive import Live
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate
import numpy as np
import polars as pl

from models.utils import log_confusion_matrix, log_roc_auc_curve


def evaluate(
    model, X_train: pl.DataFrame , X_test: pl.DataFrame,
    y_train: pl.Series, y_test: np.ndarray, y_pred: np.ndarray,
    y_pred_proba: np.ndarray, live: Live
):
    # log cross-validation for whole data (train + test)
    scoring = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
    cv_scores = cross_validate(model,
                                pl.concat([X_train, X_test]),
                                np.ravel(pl.concat([y_train, y_test])),
                                cv=5,
                                scoring=scoring,
                                n_jobs=-1)
    live.log_metric('cv_accuracy_mean', np.mean(cv_scores['test_accuracy']))
    live.log_metric('cv_f1_mean', np.mean(cv_scores['test_f1']))
    live.log_metric('cv_precision_mean', np.mean(cv_scores['test_precision']))
    live.log_metric('cv_recall_mean', np.mean(cv_scores['test_recall']))
    live.log_metric('cv_roc_auc_mean', np.mean(cv_scores['test_roc_auc']))

    live.log_metric('cv_std_accuracy', np.std(cv_scores['test_accuracy']))
    live.log_metric('cv_std_f1', np.std(cv_scores['test_f1']))
    live.log_metric('cv_std_precision', np.std(cv_scores['test_precision']))
    live.log_metric('cv_std_recall', np.std(cv_scores['test_recall']))
    live.log_metric('cv_std_roc_auc', np.std(cv_scores['test_roc_auc']))

    # log test metric
    y_test = np.ravel(y_test)
    y_pred = np.ravel(y_pred)

    live.log_metric("test_accuracy", accuracy_score(y_test, y_pred))
    live.log_metric("test_recall", recall_score(y_test, y_pred))
    live.log_metric("test_precision", precision_score(y_test, y_pred))
    live.log_metric("test_f1", f1_score(y_test, y_pred))
    live.log_metric("test_roc_auc", roc_auc_score(y_test, y_pred_proba))

    # Plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    log_confusion_matrix(live, cm, class_names=["ergonomic", "non-ergonomic"])
    log_roc_auc_curve(live, y_test, y_pred_proba)

    return cv_scores
