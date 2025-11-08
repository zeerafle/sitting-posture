import os
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
from sklearn_evaluation import plot
from sklearn.model_selection import cross_validate
import json
import numpy as np


def log_confusion_matrix(live, cm, class_names, title=None, cmap="Blues"):
    """Plots the confusion matrix."""
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 11
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        cmap=cmap,
        ax=ax,
        annot_kws={"fontsize": 15},
        xticklabels=class_names,
        yticklabels=class_names,
    )
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    if title:
        ax.set_title(title)
    # log the confusion matrix
    plt.tight_layout()
    live.log_image("confusion_matrix.png", fig)


def log_roc_auc_curve(live, y_true, y_pred_proba):
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    plot.ROC.from_raw_data(y_true, y_pred_proba, ax=ax)
    plt.tight_layout()
    live.log_image("roc_auc_curve.png", fig)


def load_data(base_path):
    train = pl.read_csv(os.path.join(base_path, "train.csv"))
    test = pl.read_csv(os.path.join(base_path, "test.csv"))

    X_train= train.select(pl.exclude("labels"))
    X_test = test.select(pl.exclude("labels"))
    y_train= train.select("labels")
    y_test = test.select("labels")

    return X_train, X_test, y_train, y_test


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, (np.ndarray,)):
            return o.tolist()
        return json.JSONEncoder.default(self, o)
