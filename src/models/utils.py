import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt


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
    live.log_image('confusion_matrix.png', fig)
    plt.show()
    plt.close(fig)


def log_roc_auc_curve(live, y_true, y_pred_proba):
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    skplt.metrics.plot_roc(y_true, y_pred_proba, ax=ax)
    live.log_image('roc_auc_curve.png', fig)
    plt.show()
    plt.close(fig)
