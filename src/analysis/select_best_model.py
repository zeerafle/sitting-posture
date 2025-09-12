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

    # Load the CSV with first column as index
    return pd.read_csv(rankings_path, index_col=0)


def select_best_model(statistical_results, model_rankings, metric, selection_criteria="mean_score"):
    """
    Select the best model based on statistical analysis results.

    Args:
        statistical_results: Statistical analysis results
        model_rankings: DataFrame with model rankings
        metric: Metric being analyzed
        selection_criteria: Criteria for selection ("mean_score", "statistical_significance", "effect_size")

    Returns:
        dict: Best model information
    """
    logger.info(f"Selecting best model based on {selection_criteria} for {metric}")

    # For Bayesian analysis, there's no omnibus test like Friedman
    # Instead, we look at pairwise model comparisons

    if selection_criteria == "mean_score":
        # Sort by mean score (descending for metrics where higher is better)
        higher_is_better = statistical_results.get('higher_is_better', True)
        if higher_is_better:
            best_model_name = model_rankings.iloc[0].name  # First row is already highest mean
        else:
            best_model_name = model_rankings.iloc[-1].name  # Last row is lowest mean

        best_model = model_rankings.loc[best_model_name]
        selection_reason = f"Highest mean {metric} ({best_model['mean']:.4f})"

    elif selection_criteria == "effect_size":
        # Select model with smallest effect size compared to the best model
        # (which is always 0 for the top model)
        best_model_name = model_rankings.iloc[0].name
        best_model = model_rankings.loc[best_model_name]

        selection_reason = "Best model by effect size comparison"

    elif selection_criteria == "statistical_significance":
        # Select top performing model that has definitive statistical advantage
        best_model_name = model_rankings.iloc[0].name
        best_model = model_rankings.loc[best_model_name]

        # Check decisions for all other models compared to this one
        significant_advantage = False
        for idx, row in model_rankings.iterrows():
            if idx == best_model_name:
                continue

            if row['decision'] == 'smaller':
                significant_advantage = True
                break

        if significant_advantage:
            selection_reason = "Statistically significant advantage over other models"
        else:
            selection_reason = "Top performing model, but without statistically significant advantage"
    else:
        raise ValueError(f"Unknown selection criteria: {selection_criteria}")

    # Prepare best model information
    best_model_info = {
        'model_name': best_model_name,
        'metric': metric,
        'selection_criteria': selection_criteria,
        'selection_reason': selection_reason,
        'performance': {
            'mean_score': float(best_model['mean']),
            'std_score': float(best_model['std']),
            'ci_lower': float(best_model['ci_lower']),
            'ci_upper': float(best_model['ci_upper'])
        },
        'statistical_analysis': {
            'omnibus_test': statistical_results['omnibus_test'],
            'posthoc_test': statistical_results['posthoc_test'],
            'significant_comparisons': []
        },
        'model_rankings': model_rankings.to_dict('records'),
        'selection_timestamp': pd.Timestamp.now().isoformat()
    }

    # Add information about which models this model is significantly better than
    for idx, row in model_rankings.iterrows():
        if idx == best_model_name:
            continue

        if row['decision'] == 'smaller':
            best_model_info['statistical_analysis']['significant_comparisons'].append({
                'compared_to': idx,
                'effect_size': float(row['effect_size']),
                'magnitude': row['magnitude'],
                'p_smaller': float(row['p_smaller']),
                'decision': row['decision']
            })

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

    print("\n=== BEST MODEL SELECTION SUMMARY ===")
    print(f"Selected Model: {model_name}")
    print(f"Metric: {metric}")
    print(f"Selection Criteria: {best_model_info['selection_criteria']}")
    print(f"Reason: {best_model_info['selection_reason']}")

    print("\nPerformance:")
    perf = best_model_info['performance']
    print(f"  Mean {metric}: {perf['mean_score']:.4f} ± {perf['std_score']:.4f}")
    print(f"  95% CI: [{perf['ci_lower']:.4f}, {perf['ci_upper']:.4f}]")

    stats = best_model_info['statistical_analysis']
    print("\nStatistical Analysis:")
    print(f"  Analysis Type: {stats['omnibus_test']}")
    print(f"  Post-hoc Test: {stats['posthoc_test']}")

    if stats['significant_comparisons']:
        print("  Significantly better than:")
        for comparison in stats['significant_comparisons']:
            print(f"    - {comparison['compared_to']}: effect_size={comparison['effect_size']:.4f} "
                 f"({comparison['magnitude']}), p_smaller={comparison['p_smaller']:.4f}")
    else:
        print("  No statistically significant advantages found")

    print("\nAll Model Rankings:")
    for i, model_data in enumerate(best_model_info['model_rankings']):
        model = model_data.get('model', model_data.get('index', f'Model {i}'))
        print(f"  {model}: mean={model_data['mean']:.4f}±{model_data['std']:.4f}, "
              f"effect_size={model_data['effect_size']:.4f} ({model_data['magnitude']})")


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
