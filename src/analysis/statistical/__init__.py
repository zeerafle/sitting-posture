"""
Statistical analysis package for model comparison.

This package provides modular tools for comparing machine learning models
using non-parametric statistical tests, leveraging the autorank library
for streamlined statistical analysis and visualization.
"""

from .data_loader import load_model_results, load_ablation_results
from .tests import (prepare_data_matrix, convert_to_dataframe,
                   run_autorank_analysis, create_result_summary)
from .output import print_results, save_results

__all__ = [
    'load_model_results',
    'load_ablation_results',
    'prepare_data_matrix',
    'convert_to_dataframe',
    'run_autorank_analysis',
    'create_result_summary',
    'print_results',
    'save_results'
]
