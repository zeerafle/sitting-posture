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
    pairwise_bayesian_signed_rank, posthoc_after_friedman, average_ranks
)
from statistical.output import print_results, save_results, print_data_summary

from statistical.plots import draw_cd_diagram


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

    # New: post-hoc control for k>=3
    parser.add_argument('--posthoc', type=str, choices=['auto', 'nemenyi', 'holm', 'shaffer'],
                        default='auto', help='Post-hoc method for k>=3 models (default: auto -> nemenyi)')
    parser.add_argument('--higher_is_better', action='store_true', default=True,
                        help='Set if higher metric value is better (default True). For losses, disable via --no-higher_is_better')
    parser.add_argument('--no-higher_is_better', dest='higher_is_better', action='store_false')

    # CD diagram
    parser.add_argument('--cd_diagram', action='store_true',
                        help='Draw and save a CD diagram of average ranks (requires Orange3)')
    parser.add_argument('--cd_test', type=str, default='nemenyi',
                        help='CD test type for Orange.compute_CD (default: nemenyi)')

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

        n_models = len(models)

        # Decide post-hoc strategy
        do_posthoc = args.force_pairwise or (p_value < args.alpha and n_models > 2) or (n_models == 2)
        post_hoc_df = pd.DataFrame()
        post_hoc_matrix = pd.DataFrame()
        post_hoc_method = None

        if do_posthoc:
            if n_models == 2:
                # Legacy: Wilcoxon with Bonferroni (no multiple pairs anyway)
                post_hoc_df = pairwise_comparisons(data_matrix, models, args.alpha)
            else:
                # k >= 3, pick method
                chosen = 'nemenyi' if args.posthoc == 'auto' else args.posthoc
                posthoc = posthoc_after_friedman(data_matrix, models, alpha=args.alpha, method=chosen)
                post_hoc_method = posthoc['method']
                post_hoc_matrix = posthoc['pmatrix']
                post_hoc_df = posthoc['pairs']

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

        # Average ranks and optional CD diagram
        avg_r = None
        cd_value = None
        cd_file = None

        if n_models >= 3:
            avg_r = average_ranks(data_matrix, higher_is_better=args.higher_is_better).tolist()

            want_cd = args.cd_diagram or (p_value < args.alpha)
            if want_cd and draw_cd_diagram is not None:
                # Choose output path
                output_path = os.path.join(os.path.dirname(dvclive_path), "analysis") if analysis_type == 'ablation' else os.path.join(dvclive_path, "analysis")
                os.makedirs(output_path, exist_ok=True)
                model_string = "_".join(args.models)
                cd_file = os.path.join(output_path, f"cd_diagram_{analysis_type}_{model_string}_{args.metric}.png")
                try:
                    cd_value = draw_cd_diagram(avg_r, models, n_datasets=len(data_matrix[0]), alpha=args.alpha, test=args.cd_test, outfile=cd_file)
                except ImportError as e:
                    print(f"CD diagram skipped: {e}")
                    cd_file = None
                    cd_value = None

        result = create_result_summary(
            statistic, p_value, kendalls_w_value, models,
            data_matrix, post_hoc_df, args.alpha,
            analysis_type, args.metric,
            bayesian_post_hoc_df=bayes_df,
            bayesian_params={'rope': args.rope, 'nsamples': args.nsamples, 'prior': args.prior}
                if not bayes_df.empty else {},
            post_hoc_matrix=post_hoc_matrix,
            post_hoc_method=post_hoc_method,
            avg_ranks=avg_r,
            cd_value=cd_value,
            cd_diagram_file=cd_file
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
