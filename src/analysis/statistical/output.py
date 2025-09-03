"""Output utilities for statistical analysis results."""
import os
import json
from typing import Dict, Any


def print_results(result: Dict[str, Any]) -> None:
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
    if result['post_hoc_results']:
        print("\nPOST-HOC PAIRWISE COMPARISONS (Wilcoxon signed-rank test):")
        print("=" * 60)
        for comparison in result['post_hoc_results']:
            status = "SIGNIFICANT" if comparison['Significant'] else "Not significant"
            print(f"{comparison['Model 1']} vs {comparison['Model 2']}: p = {comparison['P-value']:.6f} ({status})")


def save_results(result: Dict[str, Any], output_path: str, models: list, analysis_type: str, metric: str) -> None:
    """Save results to JSON file."""
    os.makedirs(output_path, exist_ok=True)

    model_string = "_".join(models)
    filename = f"friedman_{analysis_type}_{model_string}_{metric}.json"
    output_file = os.path.join(output_path, filename)

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"\nResults saved to: {output_file}")


def print_data_summary(results: Dict[str, list], models: list) -> None:
    """Print summary of loaded data."""
    print("Loaded data summary:")
    for model in models:
        scores = results.get(model, [])
        if scores:
            import numpy as np
            print(f"  {model}: {len(scores)} values, mean = {np.mean(scores):.4f}")
        else:
            print(f"  {model}: No data found")
    print()
