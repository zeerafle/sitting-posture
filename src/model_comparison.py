import json
import os
import pandas as pd
from pathlib import Path

base_path = Path("/teamspace/studios/this_studio/sitting-posture/dvclive")

def read_metrics_from_dvclive():
    """
    Read metrics from all model combinations in the dvclive folder
    and return a structured DataFrame
    """

    # Define the models available
    models = ["adaboost", "nn", "xgb"]

    data = []

    for model in models:
        metrics_file = base_path / model / "metrics.json"

        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)

                # Extract test metrics (using test_ prefix for final evaluation)
                row = {
                    'Classifier': model,
                    'acc': metrics.get('test_accuracy', 'N/A'),
                    'prec': metrics.get('test_precision', 'N/A'),
                    'rec': metrics.get('test_recall', 'N/A'),
                    'f1': metrics.get('test_f1', 'N/A'),
                    'ROC-AUC': metrics.get('test_roc_auc', 'N/A')
                }
                data.append(row)

            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error reading {metrics_file}: {e}")
                continue
        else:
            print(f"Metrics file not found: {metrics_file}")

    return pd.DataFrame(data)

def main():
    # Read all metrics
    df = read_metrics_from_dvclive()

    # Sort by classifier for better readability
    df = df.sort_values(['Classifier'])

    # Format numerical columns to percentages with 2 decimal places (without % symbol)
    numeric_cols = ['acc', 'prec', 'rec', 'f1', 'ROC-AUC']
    for col in numeric_cols:
        df[col] = df[col].apply(lambda x: f"{x*100:.2f}" if isinstance(x, (int, float)) else x)

    # Display the table
    print("Model Performance Summary")
    print("=" * 80)
    print(df.to_string(index=False))

    # Save to CSV if desired
    output_file = base_path / "model_performance_summary.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Display some summary statistics
    print("\nSummary Statistics:")
    print("-" * 40)

    # Convert back to numeric for statistics
    df_numeric = df.copy()
    for col in numeric_cols:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce') / 100

    # Best performing model per metric
    print("\nBest performing models per metric:")
    for metric in numeric_cols:
        best_idx = df_numeric[metric].idxmax()
        if pd.notna(best_idx):
            best_model = df.loc[best_idx]
            print(f"{metric.upper()}: {best_model['Classifier']} - {best_model[metric]}")

if __name__ == "__main__":
    main()
