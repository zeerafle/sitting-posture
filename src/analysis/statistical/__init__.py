"""
Statistical analysis package for model comparison.

This package provides modular tools for comparing machine learning models
using non-parametric statistical tests, specifically the Friedman test
with post-hoc pairwise comparisons.

Modules:
    data_loader: Load results from different analysis types (CV, LOSO, ablation)
    tests: Statistical tests including Friedman test and post-hoc analysis
    output: Utilities for printing and saving results
"""

from .data_loader import load_model_results, load_ablation_results
from .tests import friedman_test, kendalls_w, pairwise_comparisons
from .output import print_results, save_results

__all__ = [
    'load_model_results',
    'load_ablation_results',
    'friedman_test',
    'kendalls_w',
    'pairwise_comparisons',
    'print_results',
    'save_results'
]
