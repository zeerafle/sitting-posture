from dvclive import Live
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report
)
import polars as pl
import numpy as np

from models.utils import log_confusion_matrix, log_roc_auc_curve


def evaluate(model, X_test: pl.DataFrame, y_test: np.ndarray, live: Live):
    # Classify pose in the TEST dataset using the trained model
    y_pred_proba = model.predict(X_test.to_numpy())
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_test_ravel = np.argmax(y_test, axis=1)

    # log test metric
    live.log_metric('test_accuracy', accuracy_score(y_test_ravel, y_pred))
    live.log_metric('test_recall', recall_score(y_test_ravel, y_pred))
    live.log_metric('test_precision', precision_score(y_test_ravel, y_pred))
    live.log_metric('test_f1', f1_score(y_test_ravel, y_pred))
    live.log_metric('test_roc_auc', roc_auc_score(y_test, y_pred_proba))

    # Convert the prediction result to class name
    classes = ['ergonomic', 'non-ergonomic']
    y_pred_label = [classes[i][0] for i in y_pred]
    y_true_label = [classes[i][0] for i in y_test_ravel]

    # Plot the confusion matrix
    cm = confusion_matrix(y_test_ravel, y_pred)
    log_confusion_matrix(live, cm, class_names=['ergonomic', 'non-ergonomic'])
    log_roc_auc_curve(live, y_test_ravel, y_pred_proba)

    # Print the classification report
    print('\nClassification Report:\n', classification_report(y_true_label,
                                                              y_pred_label))
