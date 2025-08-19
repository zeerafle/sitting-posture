import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from itertools import combinations

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

# Constants for better maintainability
METRIC_KEYS_MAPPING = {
    'accuracy': ['test_accuracy', 'cv_accuracy_scores', 'accuracy_scores', 'test_score'],
    'f1': ['test_f1', 'cv_f1_scores', 'f1_scores'],
    'precision': ['test_precision', 'cv_precision_scores', 'precision_scores'],
    'recall': ['test_recall', 'cv_recall_scores', 'recall_scores'],
    'roc_auc': ['test_roc_auc', 'cv_roc_auc_scores', 'roc_auc_scores']
}

# Modular functions for each task
def load_data(file_path, metric, data_type='cv'):
    """Load data from json file, handling different formats."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        if data_type == 'loso':
            # Handle LOSO metrics format (per-subject results)
            if metric in data and isinstance(data[metric], list):
                return data[metric]
            else:
                print(f"Warning: {metric} not found in LOSO metrics: {file_path}")
                return []
        else:
            # Handle CV results format
            possible_keys = METRIC_KEYS_MAPPING.get(metric, [metric]) + ['test_score']

            for key in possible_keys:
                if key in data and isinstance(data[key], list):
                    return data[key]

            print(f"Warning: {metric} not found in CV results: {file_path}")
            print(f"Available keys: {list(data.keys())}")

            # Try to find any list-type values as fallback
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 1:
                    print(f"  Found list data in '{key}': {value[:3]}..." if len(value) > 3 else f"  Found list data in '{key}': {value}")
            return []

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {file_path}: {e}")
        return []

def load_model_results(dvclive_base_path, model_names, metric, analysis_type):
    """Load results for all models based on analysis type."""
    results = {}

    for model_name in model_names:
        if analysis_type == 'loso':
            # Load LOSO metrics
            loso_path = os.path.join(dvclive_base_path, model_name, "combined_loso", "loso_metrics.json")
            try:
                with open(loso_path, 'r') as f:
                    loso_data = json.load(f)
                # LOSO metrics should contain per-subject values for each metric
                if metric in loso_data and isinstance(loso_data[metric], list):
                    results[model_name] = loso_data[metric]
                else:
                    print(f"Warning: {metric} not found in LOSO metrics for {model_name}")
                    results[model_name] = []
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Error loading LOSO metrics for {model_name}: {e}")
                results[model_name] = []

        elif analysis_type == 'combined':
            # Load combined CV results
            cv_path = os.path.join(dvclive_base_path, model_name, "combined", "cv_results.json")
            results[model_name] = load_data(cv_path, metric)

        else:  # Individual views
            all_fold_scores = []
            for view in ['front', 'left', 'right']:
                cv_path = os.path.join(dvclive_base_path, model_name, view, "cv_results.json")
                all_fold_scores.extend(load_data(cv_path, metric))
            results[model_name] = all_fold_scores

    return results

def prepare_data_matrix(results, model_names):
    """Prepare data matrix for Friedman test, handling inconsistencies."""
    data_matrix = []

    for model_name in model_names:
        scores = results.get(model_name, [])
        valid_scores = [score for score in scores if score is not None]
        data_matrix.append(valid_scores)

    # Check if all models have the same number of results
    lengths = [len(scores) for scores in data_matrix]
    if len(set(lengths)) > 1:
        min_length = min(lengths)
        print(f"Warning: Models have different numbers of results. Using first {min_length} results for each.")
        data_matrix = [scores[:min_length] for scores in data_matrix]

    # Validate data is sufficient for testing
    if len(data_matrix) < 2:
        raise ValueError("Need at least 2 models to perform Friedman test")

    if any(len(scores) < 2 for scores in data_matrix):
        raise ValueError("Need at least 2 observations per model to perform Friedman test")

    # Convert to numpy arrays
    return [np.array(scores, dtype=float).flatten() for scores in data_matrix]

def run_friedman_test(data_matrix):
    """Run the Friedman test and return statistics."""
    try:
        statistic, p_value = friedmanchisquare(*data_matrix)
        return float(statistic), float(p_value)
    except ValueError as e:
        print(f"Error in Friedman test: {e}")
        for i, scores in enumerate(data_matrix):
            print(f"  Model data shape {i}: {scores.shape}, values: {scores}")
        raise

def calculate_kendalls_w(data_matrix):
    """Calculate Kendall's W effect size."""
    n_models = len(data_matrix)
    n_observations = len(data_matrix[0])

    # Transpose for easier rank calculation
    data = np.array(data_matrix).T

    # Calculate ranks for each observation
    ranks = np.array([rankdata(row) for row in data])

    # Sum of ranks for each model
    rank_sums = np.sum(ranks, axis=0)

    # Calculate Kendall's W
    mean_rank_sum = np.mean(rank_sums)
    sum_squared_deviations = np.sum((rank_sums - mean_rank_sum) ** 2)

    return (12 * sum_squared_deviations) / (n_observations ** 2 * (n_models ** 3 - n_models))

def perform_pairwise_comparisons(data_matrix, model_names, alpha=0.05):
    """Perform pairwise comparisons with Bonferroni correction."""
    n_models = len(model_names)
    n_comparisons = n_models * (n_models - 1) // 2
    corrected_alpha = alpha / n_comparisons
    results = []

    for i, j in combinations(range(n_models), 2):
        try:
            statistic, p_value = wilcoxon(data_matrix[i], data_matrix[j])
            significant = p_value < corrected_alpha

            results.append({
                'Model 1': model_names[i],
                'Model 2': model_names[j],
                'Statistic': float(statistic),
                'P-value': float(p_value),
                'Corrected Alpha': corrected_alpha,
                'Significant': significant
            })
        except ValueError as e:
            print(f"Warning: Wilcoxon test failed for {model_names[i]} vs {model_names[j]}: {e}")

    return pd.DataFrame(results)

def create_result_summary(statistic, p_value, kendalls_w, model_names, data_matrix, post_hoc_df, alpha, analysis_type):
    """Create structured result summary as a dictionary."""
    result = {
        'analysis_type': analysis_type,
        'friedman_statistic': float(statistic),
        'p_value': float(p_value),
        'kendalls_w': float(kendalls_w),
        'significant': bool(p_value < alpha),
        'models': model_names,
        'metric': args.metric,
        'alpha': alpha,
        'model_performance': {
            model: {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'values': scores.tolist()
            }
            for model, scores in zip(model_names, data_matrix)
        }
    }

    if not post_hoc_df.empty:
        result['post_hoc_results'] = post_hoc_df.to_dict('records')

    return result

def print_results(result):
    """Print formatted results to console."""
    print("=" * 60)
    print(f"FRIEDMAN TEST RESULTS - {result['analysis_type'].upper()} ANALYSIS")
    print("=" * 60)
    print(f"Models compared: {', '.join(result['models'])}")
    print(f"Metric: {result['metric']}")
    print(f"Number of models: {len(result['models'])}")
    print(f"Number of observations: {len(next(iter(result['model_performance'].values()))['values'])}")
    print()

    print("Test Statistics:")
    print(f"  Friedman χ² statistic: {result['friedman_statistic']:.4f}")
    print(f"  P-value: {result['p_value']:.6f}")
    print(f"  Kendall's W (effect size): {result['kendalls_w']:.4f}")
    print()

    if result['significant']:
        print(f"✓ SIGNIFICANT DIFFERENCE DETECTED (p < {result['alpha']})")
    else:
        print(f"✗ NO SIGNIFICANT DIFFERENCE (p ≥ {result['alpha']})")

    # Effect size interpretation
    w = result['kendalls_w']
    effect_size = "Small" if w < 0.1 else "Medium" if w < 0.3 else "Large"
    print(f"Effect Size: {effect_size} (Kendall's W = {w:.4f})")
    print()

    # Model performance
    print("Model Performance Summary:")
    for model, perf in result['model_performance'].items():
        print(f"  {model}: Mean = {perf['mean']:.4f}, Std = {perf['std']:.4f}")

    # Post-hoc results
    if 'post_hoc_results' in result:
        print("\nPOST-HOC PAIRWISE COMPARISONS (Wilcoxon signed-rank test):")
        print("=" * 60)
        for comparison in result['post_hoc_results']:
            status = "SIGNIFICANT" if comparison['Significant'] else "Not significant"
            print(f"{comparison['Model 1']} vs {comparison['Model 2']}: p = {comparison['P-value']:.6f} ({status})")

def main(args):
    """Main function to orchestrate the analysis."""
    # Convert relative path to absolute
    dvclive_path = os.path.abspath(os.path.join(current_dir, args.dvclive_path))

    # Determine analysis type
    if args.loso:
        analysis_type = "loso"
    elif args.combined:
        analysis_type = "combined"
    else:
        analysis_type = "individual"

    print(f"Analysis Type: {analysis_type.upper()}")
    print(f"Models: {args.models}")
    print(f"Metric: {args.metric}")
    print()

    try:
        # Load results
        results = load_model_results(dvclive_path, args.models, args.metric, analysis_type)

        # Print loaded data summary
        print("Loaded data summary:")
        for model_name in args.models:
            scores = results.get(model_name, [])
            if scores:
                print(f"  {model_name}: {len(scores)} values, mean = {np.mean(scores):.4f}")
            else:
                print(f"  {model_name}: No data found")
        print()

        # Prepare data matrix
        data_matrix = prepare_data_matrix(results, args.models)

        # Perform Friedman test
        statistic, p_value = run_friedman_test(data_matrix)

        # Calculate effect size
        kendalls_w = calculate_kendalls_w(data_matrix)

        # Perform post-hoc tests if significant
        post_hoc_df = pd.DataFrame()
        if p_value < args.alpha and len(args.models) > 2:
            post_hoc_df = perform_pairwise_comparisons(data_matrix, args.models, args.alpha)

        # Create result summary
        result = create_result_summary(
            statistic, p_value, kendalls_w, args.models,
            data_matrix, post_hoc_df, args.alpha, analysis_type
        )

        # Print results
        print_results(result)

        # Save results
        output_dir = os.path.join(dvclive_path, "analysis")
        os.makedirs(output_dir, exist_ok=True)

        model_string = "_".join(args.models)
        filename = f"friedman_{analysis_type}_{model_string}_{args.metric}.json"
        output_file = os.path.join(output_dir, filename)

        with open(output_file, 'w') as f:
            json.dump(result, f, indent=4)

        print(f"\nResults saved to: {output_file}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform Friedman test to compare model performance')
    parser.add_argument('--dvclive_path', type=str, default='../../dvclive',
                       help='Path to DVCLive logs directory')
    parser.add_argument('--models', nargs='+', required=True,
                       help='List of model names to compare')
    parser.add_argument('--metric', type=str, default='accuracy',
                       help='Metric to compare (default: accuracy)')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Significance level (default: 0.05)')

    # Analysis type options (mutually exclusive)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--combined', action='store_true',
                      help='Analyze combined view results')
    group.add_argument('--loso', action='store_true',
                      help='Analyze LOSO (Leave-One-Subject-Out) results')

    args = parser.parse_args()
    main(args)
