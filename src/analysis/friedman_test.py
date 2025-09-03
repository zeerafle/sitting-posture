#!/usr/bin/env python3
"""Friedman Test for Statistical Model Comparison - modular framework for ML model comparison."""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from statistical.data_loader import load_model_results
from statistical.tests import prepare_data_matrix, friedman_test, kendalls_w, pairwise_comparisons, create_result_summary
from statistical.output import print_results, save_results, print_data_summary


def parse_arguments():
    parser = argparse.ArgumentParser(description='Perform Friedman test to compare model performance')
    parser.add_argument('--dvclive_path', type=str, default='../../dvclive', help='Path to DVCLive logs')
    parser.add_argument('--models', nargs='+', required=True, help='Model names or config names for ablation')
    parser.add_argument('--metric', type=str, default='accuracy', help='Metric to compare')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--combined', action='store_true', help='Analyze combined view results')
    group.add_argument('--loso', action='store_true', help='Analyze LOSO results')
    group.add_argument('--ablation', action='store_true', help='Analyze ablation study results')
    return parser.parse_args()


def determine_analysis_type(args):
    if args.ablation: return 'ablation'
    elif args.loso: return 'loso'
    elif args.combined: return 'combined'
    else: return 'individual'


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

        statistic, p_value = friedman_test(data_matrix)
        kendalls_w_value = kendalls_w(data_matrix)
        post_hoc_df = pairwise_comparisons(data_matrix, models, args.alpha) if p_value < args.alpha and len(models) > 2 else pd.DataFrame()

        result = create_result_summary(statistic, p_value, kendalls_w_value, models,
                                     data_matrix, post_hoc_df, args.alpha,
                                     analysis_type, args.metric)
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
