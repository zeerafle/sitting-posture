import os
import joblib
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
from sklearn.metrics import roc_curve, auc
from models.utils import load_data


def plot_roc_curves():
    """
    Loads models and data to plot and compare ROC curves for multiple models and views.
    """
    model_names = ["adaboost", "nn", "xgb"]
    views = ["front", "left", "right"]

    # Define colors and line styles for differentiation
    colors = {'adaboost': '#1f77b4', 'nn': '#ff7f0e', 'xgb': '#2ca02c'}
    line_styles = {'front': '-', 'left': '--', 'right': ':'}

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle("ROC Curves Comparison - All Models and Views", fontsize=16)

    for view in views:
        data_path = os.path.join(f"data/processed/{view}")
        _, X_test, _, y_test = load_data(data_path)

        for model_name in model_names:
            if model_name == "xgb":
                model_path = os.path.join(
                    f"models/{model_name}/{model_name}_{view}.json"
                )
                model = xgb.XGBClassifier()
                model.load_model(model_path)
            else:
                model_path = os.path.join(
                    f"models/{model_name}/{model_name}_{view}.joblib"
                )

            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                continue

            if model_name != "xgb":
                model = joblib.load(model_path)

            # Get prediction probabilities
            y_score = model.predict_proba(X_test)[:, 1]  # Get probabilities for positive class

            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)

            # Plot ROC curve
            ax.plot(fpr, tpr,
                   color=colors[model_name],
                   linestyle=line_styles[view],
                   linewidth=2,
                   label=f'{model_name.upper()}-{view.upper()} (AUC = {roc_auc:.3f})')

    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Random')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    plots_dir = os.path.join("dvclive/plots")
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, "roc_curve_comparison.png"), dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_roc_curves()
