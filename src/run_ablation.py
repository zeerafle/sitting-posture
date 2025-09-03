import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import json
from pathlib import Path

def run_ablation(model_name, combined=False, output_dir_suffix=None):
    """
    Run ablation study analysis based on the matrix in ablation.md

    Matrix:
    | Dataset        | Keypoints-only  | Keypoints-engineered |
    | -------------- | --------------- | -------------------- |
    | Real-only      | [x]             | [x]                  |
    | Synthetic-only | [x]             | [x]                  |
    | Real-synthetic | [x]             | [x]                  |

    Args:
        model_class: Class reference to the trainer (XGBTrainer)
        combined: Whether to use combined views
        output_dir_suffix: Optional suffix for output directory
    """
    logger.info(f"Starting ablation study analysis for {model_name}")

    # Define the ablation matrix
    data_modes = ['real', 'synthetic', 'all']  # real-only, synthetic-only, real+synthetic
    feature_modes = ['keypoints_only', 'all_features']  # keypoints-only, keypoints+engineered

    # Results dictionary to store metrics
    results = {}

    # Output directory for ablation results
    base_output_dir = "dvclive/analysis"
    if output_dir_suffix:
        output_dir = Path(f"{base_output_dir}/ablation_{output_dir_suffix}")
    else:
        output_dir = Path(f"{base_output_dir}/ablation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run all combinations in the ablation matrix
    for data_mode in data_modes:
        results[data_mode] = {}

        for feature_mode in feature_modes:
            logger.info(f"Processing results for data_mode={data_mode}, feature_mode={feature_mode}")

            # Setup experiment name
            experiment_name = f"{data_mode}_{feature_mode}"
            results[data_mode][feature_mode] = {}

            # Process and analyze LOSO results for this configuration
            view = "combined_full" if combined else "front"

            # Construct the path to LOSO metrics file using the model's dvclive_dir
            # instead of hardcoding "xgb"
            loso_metrics_path = f"dvclive/{model_name}/{view}_{experiment_name}_loso/loso_metrics.json"

            if os.path.exists(loso_metrics_path):
                with open(loso_metrics_path, 'r') as f:
                    metrics = json.load(f)

                logger.info(f"Loaded metrics from {loso_metrics_path}")

                # Store the results
                results[data_mode][feature_mode] = metrics

                # Make sure the experiment output directory exists
                exp_output_dir = output_dir / experiment_name
                exp_output_dir.mkdir(exist_ok=True)

                # Save the results to the output directory
                result_path = exp_output_dir / "loso_metrics.json"
                with open(result_path, 'w') as f:
                    json.dump(metrics, f, indent=2)

                # Also create a summary file with averages
                summary = {}
                for metric in ['accuracy', 'f1', 'roc_auc']:
                    if metric in metrics:
                        summary[metric] = sum(metrics[metric]) / len(metrics[metric])

                summary_path = exp_output_dir / "summary.json"
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
            else:
                logger.error(f"Could not find metrics at {loso_metrics_path}")
                # Initialize with empty metrics to avoid errors in visualization
                results[data_mode][feature_mode] = {'accuracy': [], 'f1': [], 'roc_auc': []}

    # Generate summary of all experiments
    create_comparison_tables(results, output_dir, model_name)
    create_comparison_plots(results, output_dir, model_name)

    return results

def create_comparison_tables(results, output_dir, model_name):
    """
    Create tables comparing all experiments
    """
    # Create a summary dataframe for easy comparison
    rows = []
    for data_mode, feature_results in results.items():
        for feature_mode, metrics in feature_results.items():
            # Calculate averages
            avg_metrics = {}
            for metric in ['accuracy', 'f1', 'roc_auc']:
                if metric in metrics and metrics[metric]:
                    avg_metrics[metric] = sum(metrics[metric]) / len(metrics[metric])
                else:
                    avg_metrics[metric] = None

            rows.append({
                'Data Mode': data_mode,
                'Feature Mode': feature_mode,
                'Model': model_name,
                'Accuracy': avg_metrics['accuracy'],
                'F1 Score': avg_metrics['f1'],
                'ROC AUC': avg_metrics['roc_auc']
            })

    # Convert to dataframe
    summary_df = pd.DataFrame(rows)

    # Save as CSV
    summary_path = output_dir / "ablation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved summary table to {summary_path}")

    # Print summary
    print(f"\nAblation Study Summary for {model_name}:")
    print(summary_df.to_string(index=False, float_format="{:.4f}".format))

    # Also save as a pivot table format for easier reading
    pivot_acc = summary_df.pivot(index='Data Mode', columns='Feature Mode', values='Accuracy')
    pivot_f1 = summary_df.pivot(index='Data Mode', columns='Feature Mode', values='F1 Score')
    pivot_auc = summary_df.pivot(index='Data Mode', columns='Feature Mode', values='ROC AUC')

    # Save pivot tables
    pivot_acc.to_csv(output_dir / f"ablation_{model_name}_accuracy_pivot.csv")
    pivot_f1.to_csv(output_dir / f"ablation_{model_name}_f1_pivot.csv")
    pivot_auc.to_csv(output_dir / f"ablation_{model_name}_auc_pivot.csv")

    # Print pivot tables
    print(f"\nAccuracy by Configuration ({model_name}):")
    print(pivot_acc.to_string(float_format="{:.4f}".format))

    print(f"\nF1 Score by Configuration ({model_name}):")
    print(pivot_f1.to_string(float_format="{:.4f}".format))

    print(f"\nROC AUC by Configuration ({model_name}):")
    print(pivot_auc.to_string(float_format="{:.4f}".format))

def create_comparison_plots(results, output_dir, model_name):
    """
    Create plots comparing all experiments
    """
    # Create a dataframe for plotting
    plot_data = []
    for data_mode, feature_results in results.items():
        for feature_mode, metrics in feature_results.items():
            for metric_name in ['accuracy', 'f1', 'roc_auc']:
                if metric_name in metrics and metrics[metric_name]:
                    metric_values = metrics[metric_name]
                    for subject_idx, value in enumerate(metric_values):
                        plot_data.append({
                            'Data Mode': data_mode,
                            'Feature Mode': feature_mode,
                            'Model': model_name,
                            'Metric': metric_name.upper(),
                            'Subject': f"Subject {subject_idx+1}",
                            'Value': value
                        })

    # Skip plotting if we don't have data
    if not plot_data:
        logger.warning("No data available for plotting")
        return

    plot_df = pd.DataFrame(plot_data)

    # Create individual bar plots for each metric
    for metric in ['ACCURACY', 'F1', 'ROC_AUC']:
        plt.figure(figsize=(12, 6))
        metric_df = plot_df[plot_df['Metric'] == metric]

        if metric_df.empty:
            logger.warning(f"No data for {metric} metric")
            plt.close()
            continue

        # Calculate mean values for each configuration
        mean_df = metric_df.groupby(['Data Mode', 'Feature Mode'])['Value'].mean().reset_index()

        # Create the plot
        g = sns.barplot(x='Data Mode', y='Value', hue='Feature Mode', data=mean_df)

        # Add value labels on top of bars
        for i, bar in enumerate(g.patches):
            g.text(
                bar.get_x() + bar.get_width()/2.,
                bar.get_height() + 0.01,
                f'{bar.get_height():.4f}',
                ha='center', va='bottom', fontsize=9
            )

        plt.title(f'Ablation Study Results: {metric} ({model_name})')
        plt.ylim(0, 1.1)  # Metrics are between 0 and 1
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save the figure
        plt.savefig(output_dir / f"ablation_{model_name}_{metric.lower()}_comparison.png", dpi=300)
        plt.close()

    # Create a combined plot with all metrics
    metric_names = plot_df['Metric'].unique()
    if len(metric_names) > 0:
        plt.figure(figsize=(15, 10))

        # Calculate mean values for each configuration and metric
        mean_df = plot_df.groupby(['Data Mode', 'Feature Mode', 'Metric'])['Value'].mean().reset_index()

        # Create the plot
        g = sns.catplot(
            x='Data Mode', y='Value', hue='Feature Mode', col='Metric',
            data=mean_df, kind='bar', height=4, aspect=1.2, sharey=True
        )

        # Improve the plot
        g.set_titles("{col_name}")
        g.set_axis_labels("Data Mode", "Score")
        g.set(ylim=(0, 1))
        g.fig.suptitle(f'Ablation Study Results - {model_name}', fontsize=16, y=1.05)

        # Save the figure
        plt.tight_layout()
        plt.savefig(output_dir / f"ablation_{model_name}_metrics_comparison.png", dpi=300)
        plt.close()

    logger.info(f"Saved comparison plots to {output_dir}")

if __name__ == "__main__":
    from models.xgb.train import XGBoostTrainer

    parser = argparse.ArgumentParser(description='Run ablation study analysis')
    parser.add_argument('--combined', action='store_true',
                       help='Analyze combined views')
    parser.add_argument('--output-suffix', type=str, default=None,
                       help='Optional suffix for output directory')
    parser.add_argument('--model', type=str, default='xgb',
                       help='Model to analyze (default: xgb)')
    args = parser.parse_args()

    # Run ablation analysis
    run_ablation(args.model.lower(), combined=args.combined, output_dir_suffix=args.output_suffix)
