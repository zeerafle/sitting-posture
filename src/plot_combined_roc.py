"""
Script to combine ROC curves from all models into a single plot.
Creates a high-quality figure with large fonts and 300 DPI.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from sklearn.metrics import roc_curve, auc


def load_predictions(model_name, dvclive_base_path="dvclive"):
    """
    Load predictions and probabilities for a model.

    Args:
        model_name: Name of the model (e.g., 'adaboost', 'nn', 'xgb')
        dvclive_base_path: Base path to dvclive directory

    Returns:
        Tuple of (y_pred, y_pred_proba) or (None, None) if files don't exist
    """
    model_path = os.path.join(dvclive_base_path, model_name)
    y_pred_path = os.path.join(model_path, "y_pred.csv")
    y_pred_proba_path = os.path.join(model_path, "y_pred_proba.csv")

    if not os.path.exists(y_pred_proba_path):
        print(f"Warning: {y_pred_proba_path} not found")
        return None, None

    try:
        y_pred = np.loadtxt(y_pred_path, delimiter=",")
        y_pred_proba = np.loadtxt(y_pred_proba_path, delimiter=",")
        return y_pred, y_pred_proba
    except Exception as e:
        print(f"Error loading predictions for {model_name}: {e}")
        return None, None


def load_true_labels(data_path="data/processed"):
    """
    Load true labels from the processed data.

    Args:
        data_path: Path to processed data directory

    Returns:
        Array of true labels
    """
    train_path = os.path.join(data_path, "train.csv")
    test_path = os.path.join(data_path, "test.csv")

    train = pl.read_csv(train_path)
    test = pl.read_csv(test_path)

    # Combine train and test
    all_data = pl.concat([train, test])
    y_true = all_data.select("labels").to_numpy().ravel()

    return y_true


def plot_combined_roc_curves(output_path="dvclive/combined_roc_curves.png"):
    """
    Create a combined ROC curve plot for all models.

    Args:
        output_path: Path where to save the plot
    """
    # Model configurations
    models = {
        'adaboost': {'name': 'AdaBoost', 'color': '#1f77b4', 'linestyle': '-'},
        'nn': {'name': 'Neural Network', 'color': '#ff7f0e', 'linestyle': '--'},
        'xgb': {'name': 'XGBoost', 'color': '#2ca02c', 'linestyle': '-.'},
    }

    # Load true labels
    print("Loading true labels...")
    y_true = load_true_labels()
    print(f"Loaded {len(y_true)} labels")

    # Set up the plot with large fonts
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['figure.titlesize'] = 22

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # Plot ROC curve for each model
    for model_key, model_config in models.items():
        print(f"\nProcessing {model_config['name']}...")
        y_pred, y_pred_proba = load_predictions(model_key)

        if y_pred_proba is None:
            print(f"Skipping {model_config['name']} - no predictions found")
            continue

        # Ensure same length
        if len(y_pred_proba) != len(y_true):
            print(f"Warning: Length mismatch for {model_config['name']}")
            print(f"  y_true: {len(y_true)}, y_pred_proba: {len(y_pred_proba)}")
            continue

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Plot
        ax.plot(
            fpr, tpr,
            color=model_config['color'],
            linestyle=model_config['linestyle'],
            linewidth=2.5,
            label=f"{model_config['name']} (AUC = {roc_auc:.3f})"
        )
        print(f"  AUC: {roc_auc:.3f}")

    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier', alpha=0.5)

    # Customize plot
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title('ROC Curves - Model Comparison', fontweight='bold', pad=20)
    ax.legend(loc='lower right', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Add some padding
    plt.tight_layout()

    # Save the plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Combined ROC curves saved to: {output_path}")

    plt.close()


if __name__ == "__main__":
    import sys

    # Allow custom output path
    output_path = sys.argv[1] if len(sys.argv) > 1 else "dvclive/combined_roc_curves.png"

    print("=" * 60)
    print("Combined ROC Curve Plotter")
    print("=" * 60)

    plot_combined_roc_curves(output_path)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
