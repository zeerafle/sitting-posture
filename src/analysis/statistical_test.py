import os
import sys
import argparse
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from statistical.data_loader import load_model_results
from statistical.tests import (
    prepare_data_matrix, friedman_test, kendalls_w,
    pairwise_comparisons, create_result_summary,
    pairwise_bayesian_signed_rank
)
from statistical.output import print_results, save_results, print_data_summary


def parse_arguments():
    parser = argparse.ArgumentParser(description='Perform Friedman test to compare model performance')
    parser.add_argument('--dvclive_path', type=str, default='../../dvclive', help='Path to DVCLive logs')
    parser.add_argument('--models', nargs='+', required=True, help='Model names or config names for ablation')
    parser.add_argument('--metric', type=str, default='accuracy', help='Metric to compare')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level')

    # Analysis type
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--combined', action='store_true', help='Analyze combined view results')
    group.add_argument('--loso', action='store_true', help='Analyze LOSO results')
    group.add_argument('--ablation', action='store_true', help='Analyze ablation study results')

    # Pairwise options
    parser.add_argument('--force_pairwise', action='store_true',
                        help='Compute pairwise tests regardless of Friedman result')
    parser.add_argument('--bayesian', action='store_true',
                        help='Also compute pairwise Bayesian signed-rank tests (requires baycomp)')
    parser.add_argument('--rope', type=float, default=0.01,
                        help='ROPE width for Bayesian signed-rank test (scale of the metric; default 0.01)')
    parser.add_argument('--nsamples', type=int, default=20000,
                        help='Number of posterior samples for Bayesian signed-rank test (default 20000)')
    parser.add_argument('--prior', type=float, default=0.5,
                        help='Prior strength for Bayesian signed-rank test (default 0.5)')
    parser.add_argument('--random_state', type=int, default=None,
                        help='Random seed for Bayesian sampling')

    return parser.parse_args()


def determine_analysis_type(args):
    if args.ablation:
        return 'ablation'
    elif args.loso:
        return 'loso'
    elif args.combined:
        return 'combined'
    else:
        return 'individual'


def main():
    args = parse_arguments()
    dvclive_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.dvclive_path))
    analysis_type = determine_analysis_type(args)

    print(f"Analysis Type: {analysis_type.upper()}")
    print(f"Models/Configs: {args.models}")
    print(f"Metric: {args.metric}\n")

    try:
        if analysis_type == 'ablation':
            dvclive_path = os.path.join(dvclive_path, 'xgb')
            models = [f"combined_full_{config}_loso" for config in args.models]
        else:
            models = args.models

        results = load_model_results(dvclive_path, models, args.metric, analysis_type)
        print_data_summary(results, models)
        data_matrix = prepare_data_matrix(results, models)

        # Core omnibus test and effect size
        statistic, p_value = friedman_test(data_matrix)
        kendalls_w_value = kendalls_w(data_matrix)

        # Decide whether to run frequentist pairwise post-hoc
        n_models = len(models)
        do_pairwise = args.force_pairwise or (p_value < args.alpha and n_models > 2) or (n_models == 2)
        post_hoc_df = pairwise_comparisons(data_matrix, models, args.alpha) if do_pairwise else pd.DataFrame()

        # Optional Bayesian pairwise comparisons
        bayes_df = pd.DataFrame()
        if args.bayesian and n_models >= 2:
            try:
                bayes_df = pairwise_bayesian_signed_rank(
                    data_matrix, models,
                    rope=args.rope, nsamples=args.nsamples,
                    prior=args.prior, random_state=args.random_state
                )
            except ImportError as e:
                print(f"Bayesian tests skipped: {e}")

        result = create_result_summary(
            statistic, p_value, kendalls_w_value, models,
            data_matrix, post_hoc_df, args.alpha,
            analysis_type, args.metric,
            bayesian_post_hoc_df=bayes_df,
            bayesian_params={'rope': args.rope, 'nsamples': args.nsamples, 'prior': args.prior}
                if not bayes_df.empty else {}
        )
        print_results(result)

        output_path = os.path.join(os.path.dirname(dvclive_path), "analysis") if analysis_type == 'ablation' else os.path.join(dvclive_path, "analysis")
        save_results(result, output_path, args.models, analysis_type, args.metric)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
