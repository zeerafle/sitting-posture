import os
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json
import autorank

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from statistical.data_loader import load_model_results
from statistical.tests import prepare_data_matrix, convert_to_dataframe, run_autorank_analysis, create_result_summary
from statistical.output import print_results, save_results

def parse_arguments():
    parser = argparse.ArgumentParser(description='Perform statistical tests to compare model performance using autorank')
    parser.add_argument('--models', nargs='+', required=True, help='Model names or config names for comparison')
    parser.add_argument('--metric', type=str, default='accuracy', help='Metric to compare')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for results')

    # Analysis type argument - maintain backward compatibility
    parser.add_argument('--analysis-type', type=str, choices=['loso', 'individual', 'combined', 'ablation'],
                        default='individual', help='Analysis type')

    # Experiment type for ablation studies
    parser.add_argument('--experiment-type', type=str, default=None,
                        help='Experiment type for ablation studies (data_modes, feature_modes, full)')

    # Approach options
    parser.add_argument('--approach', type=str, choices=['frequentist', 'bayesian'],
                        default='frequentist', help='Statistical approach (default: frequentist)')

    # Bayesian options
    parser.add_argument('--rope', type=float, default=None,
                        help='ROPE width for Bayesian approach (default: None, autorank will determine automatically)')

    # Higher is better for ranking
    parser.add_argument('--higher-is-better', action='store_true', default=True,
                        help='Set if higher metric value is better (default True). For losses, use --lower-is-better')
    parser.add_argument('--lower-is-better', dest='higher_is_better', action='store_false',
                        help='Set if lower metric value is better (e.g., for loss metrics)')

    # Verbose output
    parser.add_argument('--verbose', action='store_true', help='Print verbose output from autorank')

    return parser.parse_args()

def extract_loso_metrics(data, metric):
    """Extract metrics from LOSO data structure."""
    metrics = []
    # Check if data is a dictionary with fold keys
    if isinstance(data, dict):
        for fold, fold_data in data.items():
            if isinstance(fold_data, dict) and metric in fold_data:
                metrics.append(float(fold_data[metric]))
    return metrics

def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Analysis Type: {args.analysis_type.upper()}")
    if args.analysis_type == 'ablation' and args.experiment_type:
        print(f"Experiment Type: {args.experiment_type}")
    print(f"Models: {args.models}")
    print(f"Metric: {args.metric}")
    print(f"Statistical Approach: {args.approach.upper()}")
    print(f"Output Directory: {args.output_dir}")

    try:
        # Use the existing data loader for compatibility
        dvclive_path = 'dvclive'
        results = load_model_results(dvclive_path, args.models, args.metric, args.analysis_type)

        # If results are empty, try loading directly
        if not results or all(len(scores) == 0 for scores in results.values()):
            print("Trying direct loading of fold_metrics.json files...")
            results = {}

            for model in args.models:
                if args.analysis_type == 'loso':
                    file_path = os.path.join(dvclive_path, model, 'loso', 'fold_metrics.json')
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            # Extract metric per fold
                            metrics = extract_loso_metrics(data, args.metric)
                            if metrics:
                                results[model] = metrics
                                print(f"Loaded {len(metrics)} metrics for {model}")
                    except Exception as e:
                        print(f"Warning: Could not load data from {file_path}: {e}")

                        # Try alternative file structure
                        try:
                            alt_path = os.path.join(dvclive_path, model, 'loso', 'aggregated_metrics.json')
                            with open(alt_path, 'r') as f:
                                data = json.load(f)
                                if 'fold_metrics' in data:
                                    fold_metrics = data['fold_metrics']
                                    metrics = [fold.get(args.metric) for fold in fold_metrics if fold.get(args.metric) is not None]
                                    if metrics:
                                        results[model] = metrics
                                        print(f"Loaded {len(metrics)} metrics for {model} from aggregated_metrics.json")
                        except Exception as e2:
                            print(f"Warning: Could not load alternative data for {model}: {e2}")

        # Print summary of loaded data
        print("\nData Summary:")
        for model, scores in results.items():
            if scores:
                print(f"{model}: {len(scores)} observations, mean={np.mean(scores):.4f}, std={np.std(scores):.4f}")
            else:
                print(f"{model}: No data loaded")

        if not results or all(len(scores) == 0 for scores in results.values()):
            raise ValueError("No valid data loaded for any model.")

        # Prepare data matrix
        data_matrix = prepare_data_matrix(results, args.models)

        # Convert to DataFrame for autorank
        df = convert_to_dataframe(data_matrix, args.models)

        # Run autorank analysis
        if args.approach == 'bayesian':
            # For Bayesian approach, use the specified rope or default
            rope_value = 0.01 if args.rope is None else args.rope
            autorank_result = run_autorank_analysis(
                data_matrix,
                args.models,
                alpha=args.alpha,
                approach=args.approach,
                verbose=args.verbose,
                rope=rope_value
            )
        else:
            # For frequentist approach, don't pass the rope parameter
            autorank_result = run_autorank_analysis(
                data_matrix,
                args.models,
                alpha=args.alpha,
                approach=args.approach,
                verbose=args.verbose
            )

        # Generate CD diagram
        cd_file = os.path.join(args.output_dir, f"critical_difference_diagram_{args.metric}.png")

        try:
            fig = autorank.plot_stats(autorank_result)
            fig.savefig(cd_file, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Failed to create CD diagram: {e}")
            cd_file = None

        # Create result summary
        result = create_result_summary(
            autorank_result,
            args.models,
            data_matrix,
            args.alpha,
            args.analysis_type,
            args.metric,
            higher_is_better=args.higher_is_better,
            cd_diagram_file=cd_file
        )

        # Print and save results
        print_results(result)

        # Save the model rankings
        rankings_file = os.path.join(args.output_dir, f"model_rankings_{args.metric}.csv")
        if hasattr(autorank_result, 'rankdf'):
            autorank_result.rankdf.to_csv(rankings_file)

        # Save the full statistical results
        results_file = os.path.join(args.output_dir, f"statistical_results_{args.metric}.json")
        save_results(result, args.output_dir, args.models, args.analysis_type, args.metric)

        print(f"\nResults saved to {args.output_dir}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
