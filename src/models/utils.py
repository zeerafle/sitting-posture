import os
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
from sklearn_evaluation import plot


def log_confusion_matrix(live, cm, class_names, title=None, cmap='Blues'):
    """Plots the confusion matrix."""
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 10
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    sns.heatmap(cm, annot=True, cmap=cmap,
                ax=ax, annot_kws={"fontsize": 11},
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Actual Class')
    if title:
        ax.set_title(title)
    # log the confusion matrix
    plt.tight_layout()
    live.log_image('confusion_matrix.png', fig)


def log_roc_auc_curve(live, y_true, y_pred_proba):
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    plot.ROC.from_raw_data(y_true, y_pred_proba, ax=ax)
    plt.tight_layout()
    live.log_image('roc_auc_curve.png', fig)


def load_data(base_path):
    train = pl.read_csv(os.path.join(base_path, "train.csv"))
    test = pl.read_csv(os.path.join(base_path, "test.csv"))
    val = pl.read_csv(os.path.join(base_path, "val.csv"))

    X_train = train.select(pl.exclude("labels"))
    X_val = val.select(pl.exclude("labels"))
    X_test = test.select(pl.exclude("labels"))

    y_train = train.select("labels")
    y_val = val.select("labels")
    y_test = test.select("labels")

    return X_train, y_train, X_test, y_test, X_val, y_val
