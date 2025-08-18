import os
import argparse
from pyexpat import model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc


def plot_roc_curves(combined=False):
    """
    Loads prediction data from CSV files to plot and compare ROC curves for multiple models and views.
    """
    model_names = ["adaboost", "nn", "xgb"]
    model_display_names = {"adaboost": "AdaBoost", "nn": "MLP", "xgb": "XGBoost"}

    if combined:
        views = ["combined"]
        title = "ROC Curves Comparison Across Classifiers"
        colors = {'adaboost': '#1f77b4', 'nn': '#ff7f0e', 'xgb': '#2ca02c'}
        line_styles = {'combined': '-'}
        filename = "roc_curve_combined_comparison.png"
    else:
        views = ["front", "left", "right"]
        title = "ROC Curves Comparison - All Models and Views"
        colors = {'adaboost': '#1f77b4', 'nn': '#ff7f0e', 'xgb': '#2ca02c'}
        line_styles = {'front': '-', 'left': '--', 'right': ':'}
        filename = "roc_curve_comparison.png"

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(title, fontsize=16)

    for view in views:
        for model_name in model_names:
            # Path to the predictions CSV file
            pred_file_path = os.path.join(f"dvclive/{model_name}/{view}/y_pred.csv")

            if not os.path.exists(pred_file_path):
                print(f"Predictions file not found: {pred_file_path}")
                continue

            try:
                # Load predictions from CSV
                pred_df = pd.read_csv(pred_file_path)

                # Validate required columns
                if 'y_true' not in pred_df.columns or 'y_pred_proba' not in pred_df.columns:
                    print(f"Required columns (y_true, y_pred_proba) not found in {pred_file_path}")
                    print(f"Available columns: {pred_df.columns.tolist()}")
                    continue

                y_true = pred_df['y_true'].values
                y_pred_proba = pred_df['y_pred_proba'].values  # Use probabilities for ROC

                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                roc_auc = auc(fpr, tpr)

                # Create label based on mode
                if combined:
                    label = f'{model_display_names[model_name].upper()} (AUC = {roc_auc:.3f})'
                else:
                    label = f'{model_display_names[model_name].upper()}-{view.upper()} (AUC = {roc_auc:.3f})'

                # Plot ROC curve
                ax.plot(fpr, tpr,
                       color=colors[model_name],
                       linestyle=line_styles[view],
                       linewidth=2,
                       label=label)

            except Exception as e:
                print(f"Error processing {pred_file_path}: {e}")
                continue

    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Random')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=16)
    ax.set_ylabel('True Positive Rate', fontsize=16)
    ax.legend(loc="lower right", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    plots_dir = os.path.join("dvclive/plots")
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()

    print(f"ROC curve plot saved to: dvclive/plots/{filename}")


def plot_comparison_all_modes():
    """
    Plot a comprehensive comparison showing both individual views and combined models
    """
    model_names = ["adaboost", "nn", "xgb"]
    views = ["front", "left", "right", "combined"]

    # Colors for models
    colors = {'adaboost': '#1f77b4', 'nn': '#ff7f0e', 'xgb': '#2ca02c'}
    # Line styles for views
    line_styles = {'front': '-', 'left': '--', 'right': ':', 'combined': '-'}
    # Alpha values to distinguish combined from individual
    alphas = {'front': 0.7, 'left': 0.7, 'right': 0.7, 'combined': 1.0}
    # Line widths
    line_widths = {'front': 1.5, 'left': 1.5, 'right': 1.5, 'combined': 3}

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.suptitle("ROC Curves Comparison - Individual Views vs Combined Models", fontsize=16)

    for view in views:
        for model_name in model_names:
            # Path to the predictions CSV file
            pred_file_path = os.path.join(f"dvclive/{model_name}/{view}/y_pred.csv")

            if not os.path.exists(pred_file_path):
                continue

            try:
                # Load predictions from CSV
                pred_df = pd.read_csv(pred_file_path)

                # Validate required columns
                if 'y_true' not in pred_df.columns or 'y_pred_proba' not in pred_df.columns:
                    continue

                y_true = pred_df['y_true'].values
                y_pred_proba = pred_df['y_pred_proba'].values

                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                roc_auc = auc(fpr, tpr)

                # Create label
                if view == "combined":
                    label = f'{model_name.upper()}-COMBINED (AUC = {roc_auc:.3f})'
                else:
                    label = f'{model_name.upper()}-{view.upper()} (AUC = {roc_auc:.3f})'

                # Plot ROC curve
                ax.plot(fpr, tpr,
                       color=colors[model_name],
                       linestyle=line_styles[view],
                       linewidth=line_widths[view],
                       alpha=alphas[view],
                       label=label)

            except Exception as e:
                print(f"Error processing {pred_file_path}: {e}")
                continue

    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Random')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    plots_dir = os.path.join("dvclive/plots")
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, "roc_curve_comprehensive_comparison.png"), dpi=300, bbox_inches='tight')
    plt.show()

    print("Comprehensive ROC curve plot saved to: dvclive/plots/roc_curve_comprehensive_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ROC curve visualizations")
    parser.add_argument("--combined", action="store_true", help="Plot only combined view models")
    parser.add_argument("--comprehensive", action="store_true", help="Plot comprehensive comparison of all modes")
    args = parser.parse_args()

    if args.comprehensive:
        plot_comparison_all_modes()
    else:
        plot_roc_curves(combined=args.combined)
