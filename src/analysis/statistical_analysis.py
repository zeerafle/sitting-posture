import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import statistical analysis modules
from statistical.data_loader import load_model_results
from statistical.tests import (
    prepare_data_matrix, friedman_test, kendalls_w,
    pairwise_comparisons, create_result_summary,
    pairwise_bayesian_signed_rank, posthoc_after_friedman, average_ranks
)
from statistical.output import print_results, save_results, print_data_summary
from statistical.plots import draw_cd_diagram


def load_loso_results(models, metric, dvclive_path="dvclive"):
    """
    Load LOSO fold-wise results for statistical analysis.

    Args:
        models: List of model names
        metric: Metric to analyze
        dvclive_path: Path to DVCLive logs

    Returns:
        pd.DataFrame: DataFrame with columns [model, fold, metric_value]
    """
    results = []

    for model in models:
        fold_metrics_path = os.path.join(dvclive_path, model, "loso", "fold_metrics.json")

        if not os.path.exists(fold_metrics_path):
            logger.warning(f"Fold metrics not found for {model}: {fold_metrics_path}")
            continue

        with open(fold_metrics_path, 'r') as f:
            fold_data = json.load(f)

        if metric not in fold_data:
            logger.warning(f"Metric '{metric}' not found in {model} results")
            continue

        metric_values = fold_data[metric]
        for fold_idx, value in enumerate(metric_values, 1):
            results.append({
                'model': model,
                'fold': fold_idx,
                'metric_value': value
            })

    return pd.DataFrame(results)


def load_ablation_results(models, metric, dvclive_path="dvclive"):
    """
    Load ablation experiment results for statistical analysis.

    Args:
        models: List of model configurations (e.g., ['real_keypoints_only', 'all_all_features'])
        metric: Metric to analyze
        dvclive_path: Path to DVCLive logs

    Returns:
        pd.DataFrame: DataFrame with columns [model, fold, metric_value]
    """
    results = []

    for model_config in models:
        fold_metrics_path = os.path.join(dvclive_path, "ablation", model_config, "fold_metrics.json")

        if not os.path.exists(fold_metrics_path):
            logger.warning(f"Ablation metrics not found for {model_config}: {fold_metrics_path}")
            continue

        with open(fold_metrics_path, 'r') as f:
            fold_data = json.load(f)

        if metric not in fold_data:
            logger.warning(f"Metric '{metric}' not found in {model_config} results")
            continue

        metric_values = fold_data[metric]
        for fold_idx, value in enumerate(metric_values, 1):
            results.append({
                'model': model_config,
                'fold': fold_idx,
                'metric_value': value
            })

    return pd.DataFrame(results)


def perform_statistical_analysis(df, models, metric, output_dir, alpha=0.05):
    """
    Perform comprehensive statistical analysis on model results.

    Args:
        df: DataFrame with columns [model, fold, metric_value]
        models: List of model names
        metric: Metric being analyzed
        output_dir: Directory to save results
        alpha: Significance level

    Returns:
        dict: Statistical analysis results
    """
    logger.info(f"Performing statistical analysis for {metric} on {len(models)} models")

    # Prepare data matrix for analysis
    data_matrix = prepare_data_matrix(df, models, metric)

    if data_matrix is None or data_matrix.empty:
        logger.error("Failed to prepare data matrix")
        return None

    logger.info(f"Data matrix shape: {data_matrix.shape}")
    print_data_summary(data_matrix, models)

    # Perform Friedman test
    friedman_result = friedman_test(data_matrix, alpha)
    logger.info(f"Friedman test p-value: {friedman_result['p_value']:.6f}")

    # Calculate Kendall's W
    kendalls_result = kendalls_w(data_matrix)
    logger.info(f"Kendall's W: {kendalls_result['kendalls_w']:.4f}")

    # Calculate average ranks
    ranks_result = average_ranks(data_matrix)

    # Perform pairwise comparisons
    pairwise_result = None
    bayesian_result = None
    posthoc_result = None

    # Always perform pairwise tests for detailed comparison
    logger.info("Performing pairwise comparisons")
    pairwise_result = pairwise_comparisons(data_matrix, alpha)

    # Bayesian signed-rank test
    try:
        logger.info("Performing Bayesian signed-rank tests")
        bayesian_result = pairwise_bayesian_signed_rank(
            data_matrix,
            rope=0.01,
            nsamples=20000,
            prior=0.5,
            random_state=42
        )
    except Exception as e:
        logger.warning(f"Bayesian analysis failed: {e}")

    # Post-hoc analysis if significant
    if friedman_result['significant']:
        logger.info("Performing post-hoc analysis (Nemenyi test)")
        posthoc_result = posthoc_after_friedman(data_matrix, alpha, method='nemenyi')

    # Compile results
    results = {
        'metric': metric,
        'models': models,
        'data_summary': {
            'n_models': len(models),
            'n_folds': len(data_matrix),
            'data_matrix': data_matrix.to_dict('records')
        },
        'friedman_test': friedman_result,
        'kendalls_w': kendalls_result,
        'average_ranks': ranks_result,
        'pairwise_comparisons': pairwise_result,
        'bayesian_comparisons': bayesian_result,
        'posthoc_analysis': posthoc_result,
        'analysis_timestamp': pd.Timestamp.now().isoformat()
    }

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    # Save JSON results
    results_path = os.path.join(output_dir, f"statistical_results_{metric}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Saved statistical results to {results_path}")

    # Create and save critical difference diagram
    try:
        cd_diagram_path = os.path.join(output_dir, f"critical_difference_diagram_{metric}.png")
        draw_cd_diagram(
            ranks_result['average_ranks'],
            models,
            cd_diagram_path,
            alpha=alpha,
            title=f"Critical Difference Diagram - {metric.capitalize()}"
        )
        logger.info(f"Saved critical difference diagram to {cd_diagram_path}")
    except Exception as e:
        logger.warning(f"Failed to create CD diagram: {e}")

    # Create model rankings CSV
    rankings_df = pd.DataFrame({
        'model': models,
        'average_rank': [ranks_result['average_ranks'][model] for model in models],
        'mean_score': [data_matrix[model].mean() for model in models],
        'std_score': [data_matrix[model].std() for model in models]
    }).sort_values('average_rank')

    rankings_path = os.path.join(output_dir, f"model_rankings_{metric}.csv")
    rankings_df.to_csv(rankings_path, index=False)
    logger.info(f"Saved model rankings to {rankings_path}")

    # Print summary
    print_results(results)

    return results


def create_summary_plots(df, models, metric, output_dir):
    """Create summary plots for the analysis."""
    os.makedirs(output_dir, exist_ok=True)

    # Box plot
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='model', y='metric_value')
    plt.title(f'{metric.capitalize()} Distribution by Model')
    plt.xlabel('Model')
    plt.ylabel(f'{metric.capitalize()}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    boxplot_path = os.path.join(output_dir, f"boxplot_{metric}.png")
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved boxplot to {boxplot_path}")

    # Performance comparison plot
    summary_stats = df.groupby('model')['metric_value'].agg(['mean', 'std']).reset_index()

    plt.figure(figsize=(12, 8))
    x_pos = range(len(summary_stats))
    plt.errorbar(x_pos, summary_stats['mean'], yerr=summary_stats['std'],
                fmt='o', capsize=5, capthick=2, markersize=8)
    plt.xlabel('Model')
    plt.ylabel(f'Mean {metric.capitalize()} ± Std')
    plt.title(f'Model Performance Comparison - {metric.capitalize()}')
    plt.xticks(x_pos, summary_stats['model'], rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    performance_path = os.path.join(output_dir, f"performance_comparison_{metric}.png")
    plt.savefig(performance_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved performance comparison to {performance_path}")


def main():
    parser = argparse.ArgumentParser(description='Perform statistical analysis on model results')
    parser.add_argument('--models', nargs='+', required=True,
                       help='Model names or configurations to analyze')
    parser.add_argument('--metric', type=str, required=True,
                       help='Metric to analyze (accuracy, f1, etc.)')
    parser.add_argument('--analysis-type', type=str, choices=['loso', 'ablation'], required=True,
                       help='Type of analysis: loso or ablation')
    parser.add_argument('--experiment-type', type=str,
                       choices=['data_modes', 'feature_modes', 'full'], default=None,
                       help='For ablation: type of experiment being analyzed')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save analysis results')
    parser.add_argument('--dvclive-path', type=str, default='dvclive',
                       help='Path to DVCLive logs')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Significance level for statistical tests')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load results based on analysis type
    if args.analysis_type == 'loso':
        df = load_loso_results(args.models, args.metric, args.dvclive_path)
    elif args.analysis_type == 'ablation':
        df = load_ablation_results(args.models, args.metric, args.dvclive_path)
    else:
        raise ValueError(f"Unknown analysis type: {args.analysis_type}")

    if df.empty:
        logger.error("No data loaded for analysis")
        return

    logger.info(f"Loaded {len(df)} data points for {len(args.models)} models")

    # Verify we have data for all models
    missing_models = set(args.models) - set(df['model'].unique())
    if missing_models:
        logger.warning(f"Missing data for models: {missing_models}")

    available_models = [m for m in args.models if m in df['model'].unique()]

    if len(available_models) < 2:
        logger.error("Need at least 2 models with data for statistical analysis")
        return

    # Perform statistical analysis
    results = perform_statistical_analysis(
        df, available_models, args.metric, args.output_dir, args.alpha
    )

    if results:
        # Create summary plots
        create_summary_plots(df, available_models, args.metric, args.output_dir)

        logger.success(f"Statistical analysis completed successfully")
        logger.info(f"Results saved to: {args.output_dir}")

        # Print key findings
        friedman_p = results['friedman_test']['p_value']
        kendalls_w = results['kendalls_w']['kendalls_w']

        print(f"\n=== KEY FINDINGS ===")
        print(f"Metric: {args.metric}")
        print(f"Models analyzed: {len(available_models)}")
        print(f"Friedman test p-value: {friedman_p:.6f}")
        print(f"Significant differences: {'Yes' if friedman_p < args.alpha else 'No'}")
        print(f"Kendall's W (effect size): {kendalls_w:.4f}")

        # Show best performing model
        rankings = results['average_ranks']['average_ranks']
        best_model = min(rankings.keys(), key=lambda k: rankings[k])
        print(f"Best performing model: {best_model} (rank: {rankings[best_model]:.2f})")
    else:
        logger.error("Statistical analysis failed")


if __name__ == "__main__":
    main()
