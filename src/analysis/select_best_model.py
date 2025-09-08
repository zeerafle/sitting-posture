import os
import json
import argparse
import pandas as pd
from loguru import logger


def load_statistical_results(analysis_dir, metric):
    """
    Load statistical analysis results for a specific metric.

    Args:
        analysis_dir: Directory containing statistical analysis results
        metric: Metric to analyze

    Returns:
        dict: Statistical analysis results
    """
    results_path = os.path.join(analysis_dir, f"statistical_results_{metric}.json")

    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Statistical results not found: {results_path}")

    with open(results_path, 'r') as f:
        results = json.load(f)

    return results


def load_model_rankings(analysis_dir, metric):
    """
    Load model rankings CSV for a specific metric.

    Args:
        analysis_dir: Directory containing analysis results
        metric: Metric to analyze

    Returns:
        pd.DataFrame: Model rankings
    """
    rankings_path = os.path.join(analysis_dir, f"model_rankings_{metric}.csv")

    if not os.path.exists(rankings_path):
        raise FileNotFoundError(f"Model rankings not found: {rankings_path}")

    return pd.read_csv(rankings_path)


def select_best_model(statistical_results, model_rankings, metric, selection_criteria="average_rank"):
    """
    Select the best model based on statistical analysis results.

    Args:
        statistical_results: Statistical analysis results
        model_rankings: DataFrame with model rankings
        metric: Metric being analyzed
        selection_criteria: Criteria for selection ("average_rank", "mean_score", "statistical_significance")

    Returns:
        dict: Best model information
    """
    logger.info(f"Selecting best model based on {selection_criteria} for {metric}")

    # Check if there are significant differences
    friedman_significant = statistical_results['friedman_test']['significant']

    if selection_criteria == "average_rank":
        # Select model with best (lowest) average rank
        best_model_idx = model_rankings['average_rank'].idxmin()
        best_model = model_rankings.loc[best_model_idx]

        selection_reason = f"Best average rank ({best_model['average_rank']:.3f})"

    elif selection_criteria == "mean_score":
        # Select model with highest mean score
        best_model_idx = model_rankings['mean_score'].idxmax()
        best_model = model_rankings.loc[best_model_idx]

        selection_reason = f"Highest mean {metric} ({best_model['mean_score']:.4f})"

    elif selection_criteria == "statistical_significance":
        # If there are significant differences, use average rank
        # Otherwise, use the model with highest mean score
        if friedman_significant:
            best_model_idx = model_rankings['average_rank'].idxmin()
            best_model = model_rankings.loc[best_model_idx]
            selection_reason = f"Statistically significant differences found, selected by rank ({best_model['average_rank']:.3f})"
        else:
            best_model_idx = model_rankings['mean_score'].idxmax()
            best_model = model_rankings.loc[best_model_idx]
            selection_reason = f"No significant differences, selected by mean score ({best_model['mean_score']:.4f})"
    else:
        raise ValueError(f"Unknown selection criteria: {selection_criteria}")

    # Get additional information
    model_name = best_model['model']

    # Check for statistical significance in pairwise comparisons
    pairwise_results = statistical_results.get('pairwise_comparisons', {})
    significantly_better_than = []

    if pairwise_results and 'comparisons' in pairwise_results:
        for comparison in pairwise_results['comparisons']:
            if comparison['model1'] == model_name and comparison['significant']:
                significantly_better_than.append(comparison['model2'])
            elif comparison['model2'] == model_name and comparison['significant']:
                significantly_better_than.append(comparison['model1'])

    # Prepare best model information
    best_model_info = {
        'model_name': model_name,
        'metric': metric,
        'selection_criteria': selection_criteria,
        'selection_reason': selection_reason,
        'performance': {
            'average_rank': float(best_model['average_rank']),
            'mean_score': float(best_model['mean_score']),
            'std_score': float(best_model['std_score'])
        },
        'statistical_analysis': {
            'friedman_test_significant': friedman_significant,
            'friedman_p_value': statistical_results['friedman_test']['p_value'],
            'kendalls_w': statistical_results['kendalls_w']['kendalls_w'],
            'significantly_better_than': significantly_better_than
        },
        'model_rankings': model_rankings.to_dict('records'),
        'selection_timestamp': pd.Timestamp.now().isoformat()
    }

    return best_model_info


def save_best_model_config(best_model_info, output_path):
    """
    Save best model configuration to JSON file.

    Args:
        best_model_info: Best model information
        output_path: Path to save the configuration
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(best_model_info, f, indent=2, default=str)

    logger.info(f"Saved best model configuration to {output_path}")


def print_best_model_summary(best_model_info):
    """Print a summary of the best model selection."""
    model_name = best_model_info['model_name']
    metric = best_model_info['metric']

    print(f"\n=== BEST MODEL SELECTION SUMMARY ===")
    print(f"Selected Model: {model_name}")
    print(f"Metric: {metric}")
    print(f"Selection Criteria: {best_model_info['selection_criteria']}")
    print(f"Reason: {best_model_info['selection_reason']}")

    print(f"\nPerformance:")
    perf = best_model_info['performance']
    print(f"  Average Rank: {perf['average_rank']:.3f}")
    print(f"  Mean {metric}: {perf['mean_score']:.4f} ± {perf['std_score']:.4f}")

    stats = best_model_info['statistical_analysis']
    print(f"\nStatistical Analysis:")
    print(f"  Friedman Test Significant: {stats['friedman_test_significant']}")
    print(f"  Friedman p-value: {stats['friedman_p_value']:.6f}")
    print(f"  Kendall's W: {stats['kendalls_w']:.4f}")

    if stats['significantly_better_than']:
        print(f"  Significantly better than: {', '.join(stats['significantly_better_than'])}")
    else:
        print(f"  No statistically significant differences found")

    print(f"\nAll Model Rankings:")
    for rank_info in best_model_info['model_rankings']:
        print(f"  {rank_info['model']}: rank={rank_info['average_rank']:.3f}, "
              f"score={rank_info['mean_score']:.4f}±{rank_info['std_score']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Select best model from statistical analysis results')
    parser.add_argument('--metric', type=str, required=True,
                       help='Metric to use for model selection (e.g., accuracy, f1)')
    parser.add_argument('--analysis-dir', type=str, required=True,
                       help='Directory containing statistical analysis results')
    parser.add_argument('--selection-criteria', type=str,
                       choices=['average_rank', 'mean_score', 'statistical_significance'],
                       default='statistical_significance',
                       help='Criteria for selecting best model')
    parser.add_argument('--output-path', type=str, default=None,
                       help='Path to save best model configuration (default: analysis-dir/best_model.json)')

    args = parser.parse_args()

    # Set default output path if not provided
    if args.output_path is None:
        args.output_path = os.path.join(os.path.dirname(args.analysis_dir), "best_model.json")

    try:
        # Load statistical results and model rankings
        logger.info(f"Loading statistical analysis results from {args.analysis_dir}")
        statistical_results = load_statistical_results(args.analysis_dir, args.metric)
        model_rankings = load_model_rankings(args.analysis_dir, args.metric)

        logger.info(f"Found results for {len(model_rankings)} models")

        # Select best model
        best_model_info = select_best_model(
            statistical_results,
            model_rankings,
            args.metric,
            args.selection_criteria
        )

        # Save best model configuration
        save_best_model_config(best_model_info, args.output_path)

        # Print summary
        print_best_model_summary(best_model_info)

        logger.success(f"Best model selection completed: {best_model_info['model_name']}")

    except Exception as e:
        logger.error(f"Error in best model selection: {e}")
        raise


if __name__ == "__main__":
    main()
