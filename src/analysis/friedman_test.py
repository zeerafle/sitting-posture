import os
import sys
import json
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
from scipy.stats import rankdata
from itertools import combinations
import argparse

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)


def load_model_results(dvclive_base_path, model_names, views, metric="accuracy"):
    """
    Load results from DVCLive logs for multiple models and views.

    Args:
        dvclive_base_path: Base path to DVCLive logs
        model_names: List of model names to compare
        views: List of views (e.g., ['front', 'left', 'right'])
        metric: Metric to compare (default: 'accuracy')

    Returns:
        Dictionary with model results
    """
    metric = "test_" + metric.lower()  # Ensure metric is prefixed with 'test_'
    results = {}

    for model_name in model_names:
        model_results = []
        for view in views:
            cv_results_path = os.path.join(
                dvclive_base_path, model_name, view, "cv_results.json"
            )

            if os.path.exists(cv_results_path):
                with open(cv_results_path, 'r') as f:
                    cv_data = json.load(f)

                # Extract the metric value
                if metric in cv_data:
                    model_results.append(cv_data[metric])
                else:
                    print(f"Warning: {metric} not found in {cv_results_path}")
                    model_results.append(None)
            else:
                print(f"Warning: Results file not found: {cv_results_path}")
                model_results.append(None)

        results[model_name] = model_results

    return results


def calculate_friedman_test(results, model_names, views):
    """
    Perform Friedman test to compare multiple models across different views.

    Args:
        results: Dictionary with model results
        model_names: List of model names
        views: List of views

    Returns:
        Friedman test statistic and p-value
    """
    # Prepare data for Friedman test
    data_matrix = []

    for model_name in model_names:
        model_scores = results[model_name]
        # Remove None values and corresponding views
        valid_scores = [score for score in model_scores if score is not None]
        data_matrix.append(valid_scores)

    # Check if all models have the same number of valid results
    lengths = [len(scores) for scores in data_matrix]
    if len(set(lengths)) > 1:
        min_length = min(lengths)
        print(f"Warning: Models have different numbers of results. Using first {min_length} results for each.")
        data_matrix = [scores[:min_length] for scores in data_matrix]

    if len(data_matrix) < 2:
        raise ValueError("Need at least 2 models to perform Friedman test")

    if len(data_matrix[0]) < 2:
        raise ValueError("Need at least 2 observations (views) to perform Friedman test")

    # Convert to numpy arrays and ensure they're 1D
    data_matrix = [np.array(scores, dtype=float).flatten() for scores in data_matrix]

    # Perform Friedman test with proper error handling
    try:
        statistic, p_value = friedmanchisquare(*data_matrix)
        # Ensure we're returning scalar values, not arrays
        return float(statistic), float(p_value), data_matrix
    except ValueError as e:
        print(f"Error in Friedman test: {e}")
        print("Data matrix shape:")
        for i, scores in enumerate(data_matrix):
            print(f"  Model {model_names[i]}: {scores}")
        raise


def calculate_effect_size(data_matrix):
    """
    Calculate Kendall's W (effect size) for Friedman test.

    Args:
        data_matrix: List of lists containing model scores

    Returns:
        Kendall's W coefficient
    """
    n_models = len(data_matrix)
    n_views = len(data_matrix[0])

    # Convert to numpy array for easier manipulation
    data = np.array(data_matrix).T  # Transpose so rows are views, columns are models

    # Calculate ranks for each view (row)
    ranks = np.array([rankdata(row) for row in data])

    # Sum of ranks for each model
    rank_sums = np.sum(ranks, axis=0)

    # Calculate Kendall's W
    mean_rank_sum = np.mean(rank_sums)
    sum_squared_deviations = np.sum((rank_sums - mean_rank_sum) ** 2)

    kendalls_w = (12 * sum_squared_deviations) / (n_views ** 2 * (n_models ** 3 - n_models))

    return kendalls_w


def post_hoc_analysis(data_matrix, model_names, alpha=0.05):
    """
    Perform post-hoc pairwise comparisons using Wilcoxon signed-rank test.

    Args:
        data_matrix: List of lists containing model scores
        model_names: List of model names
        alpha: Significance level

    Returns:
        DataFrame with pairwise comparison results
    """
    from scipy.stats import wilcoxon

    n_models = len(model_names)
    n_comparisons = n_models * (n_models - 1) // 2

    # Bonferroni correction
    corrected_alpha = alpha / n_comparisons

    results = []

    for i, j in combinations(range(n_models), 2):
        model1_scores = data_matrix[i]
        model2_scores = data_matrix[j]

        # Perform Wilcoxon signed-rank test
        try:
            statistic, p_value = wilcoxon(model1_scores, model2_scores)
            significant = p_value < corrected_alpha

            results.append({
                'Model 1': model_names[i],
                'Model 2': model_names[j],
                'Statistic': statistic,
                'P-value': p_value,
                'Corrected Alpha': corrected_alpha,
                'Significant': significant
            })
        except ValueError as e:
            print(f"Warning: Could not perform Wilcoxon test for {model_names[i]} vs {model_names[j]}: {e}")

    return pd.DataFrame(results)


def print_results(statistic, p_value, kendalls_w, model_names, data_matrix, post_hoc_df, alpha=0.05):
    """Print formatted results of the Friedman test analysis."""

    print("=" * 60)
    print("FRIEDMAN TEST RESULTS")
    print("=" * 60)
    print(f"Models compared: {', '.join(model_names)}")
    print(f"Number of models: {len(model_names)}")
    print(f"Number of observations (views): {len(data_matrix[0])}")
    print()

    print("Test Statistics:")
    print(f"  Friedman χ² statistic: {statistic:.4f}")
    print(f"  P-value: {p_value:.6f}")
    print(f"  Kendall's W (effect size): {kendalls_w:.4f}")
    print()

    if p_value < alpha:
        print(f"✓ SIGNIFICANT DIFFERENCE DETECTED (p < {alpha})")
        print("There is a statistically significant difference between the models.")
    else:
        print(f"✗ NO SIGNIFICANT DIFFERENCE (p ≥ {alpha})")
        print("No statistically significant difference between the models.")

    print()
    print("Effect Size Interpretation (Kendall's W):")
    if kendalls_w < 0.1:
        print("  Small effect")
    elif kendalls_w < 0.3:
        print("  Medium effect")
    else:
        print("  Large effect")

    print()
    print("Model Performance Summary:")
    for i, model_name in enumerate(model_names):
        scores = data_matrix[i]
        print(f"  {model_name}: Mean = {np.mean(scores):.4f}, Std = {np.std(scores):.4f}")

    if not post_hoc_df.empty:
        print()
        print("POST-HOC PAIRWISE COMPARISONS (Wilcoxon signed-rank test):")
        print("=" * 60)
        for _, row in post_hoc_df.iterrows():
            status = "SIGNIFICANT" if row['Significant'] else "Not significant"
            print(f"{row['Model 1']} vs {row['Model 2']}: p = {row['P-value']:.6f} ({status})")


def main():
    parser = argparse.ArgumentParser(description='Perform Friedman test to compare model performance')
    parser.add_argument('--dvclive_path', type=str, default='../../dvclive',
                       help='Path to DVCLive logs directory')
    parser.add_argument('--models', nargs='+', required=True,
                       help='List of model names to compare')
    parser.add_argument('--views', nargs='+', default=['front', 'left', 'right'],
                       help='List of views to compare across')
    parser.add_argument('--metric', type=str, default='accuracy',
                       help='Metric to compare (default: accuracy)')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Significance level (default: 0.05)')
    parser.add_argument('--output', type=str, help='Output file to save results')

    args = parser.parse_args()

    # Convert relative path to absolute
    dvclive_path = os.path.abspath(os.path.join(current_dir, args.dvclive_path))

    try:
        # Load results
        print(f"Loading results for models: {args.models}")
        print(f"Views: {args.views}")
        print(f"Metric: {args.metric}")
        print()

        results = load_model_results(dvclive_path, args.models, args.views, args.metric)

        # Perform Friedman test
        statistic, p_value, data_matrix = calculate_friedman_test(results, args.models, args.views)

        # Calculate effect size
        kendalls_w = calculate_effect_size(data_matrix)

        # Post-hoc analysis if significant
        post_hoc_df = pd.DataFrame()
        if p_value < args.alpha and len(args.models) > 2:
            post_hoc_df = post_hoc_analysis(data_matrix, args.models, args.alpha)

        # Print results
        print_results(statistic, p_value, kendalls_w, args.models, data_matrix, post_hoc_df, args.alpha)

        # Save results if output file specified
        if args.output:
            output_data = {
                'friedman_statistic': float(statistic),
                'p_value': float(p_value),
                'kendalls_w': float(kendalls_w),
                'significant': bool(p_value < args.alpha),
                'models': args.models,
                'views': args.views,
                'metric': args.metric,
                'alpha': args.alpha,
                'model_scores': {model: scores for model, scores in zip(args.models, data_matrix)}
            }

            if not post_hoc_df.empty:
                output_data['post_hoc_results'] = post_hoc_df.to_dict('records')

            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=4)

            print(f"\nResults saved to: {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
