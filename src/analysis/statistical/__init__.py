"""
Statistical analysis package for model comparison.

This package provides modular tools for comparing machine learning models
using non-parametric statistical tests, specifically the Friedman test
with post-hoc pairwise comparisons.
"""

from .data_loader import load_model_results, load_ablation_results
from .tests import (friedman_test, kendalls_w, pairwise_comparisons,
                    pairwise_bayesian_signed_rank, posthoc_after_friedman,
                    average_ranks, create_result_summary)
from .output import print_results, save_results

__all__ = [
    'load_model_results',
    'load_ablation_results',
    'friedman_test',
    'kendalls_w',
    'pairwise_comparisons',
    'pairwise_bayesian_signed_rank',
    'posthoc_after_friedman',
    'average_ranks',
    'create_result_summary',
    'print_results',
    'save_results'
]
