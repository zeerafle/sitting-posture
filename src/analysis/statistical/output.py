import os
import json
from typing import Dict, Any


def print_results(result: Dict[str, Any]) -> None:
    """Print formatted results to console."""
    print("=" * 60)
    print(f"STATISTICAL TEST RESULTS - {result['analysis_type'].upper()} ANALYSIS")
    print("=" * 60)
    print(f"Models compared: {', '.join(result['models'])}")
    print(f"Metric: {result['metric']}")
    print(f"Number of models: {len(result['models'])}")
    print(f"Number of observations: {len(next(iter(result['model_performance'].values()))['values'])}")
    print()

    print("Test Statistics:")
    # Use omnibus_test info if available
    omnibus_test = result.get('omnibus_test', 'friedman')
    if omnibus_test == 'bayes':
        print(f"  Test: {omnibus_test.capitalize()}")
    else:
        # For frequentist approaches
        p_value = result.get('p_value', 0.0)
        print(f"  Test: {omnibus_test.capitalize()}")
        print(f"  P-value: {p_value:.6f}")
    print()

    if result.get('significant') is not None:
        if result['significant']:
            print(f"✓ SIGNIFICANT DIFFERENCE DETECTED (p < {result['alpha']})")
        else:
            print(f"✗ NO SIGNIFICANT DIFFERENCE (p ≥ {result['alpha']})")
    print()

    # Model performance
    print("Model Performance Summary:")
    for model, perf in result['model_performance'].items():
        # Try to include both mean and median if available
        if 'mean' in perf and 'median' in perf:
            print(f"  {model}: Mean = {perf['mean']:.4f}, Median = {perf['median']:.4f}, Std = {perf['std']:.4f}")
        elif 'mean' in perf:
            print(f"  {model}: Mean = {perf['mean']:.4f}, Std = {perf['std']:.4f}")
        elif 'median' in perf:
            print(f"  {model}: Median = {perf['median']:.4f}, MAD = {perf.get('mad', 0.0):.4f}")

    # Average ranks and CD info
    avg_ranks = result.get('average_ranks', {})
    if avg_ranks:
        print("\nAverage Ranks (lower is better):")
        for m, r in avg_ranks.items():
            print(f"  {m}: {r:.3f}")
    cd = result.get('cd_value', None)
    if cd is not None:
        print(f"\nCritical Difference (CD): {cd:.3f}")
        cd_file = result.get('cd_diagram_file')
        if cd_file:
            print(f"CD diagram saved to: {cd_file}")

    # Post-hoc results (frequentist)
    if result.get('post_hoc_results'):
        print("\nPOST-HOC COMPARISONS:")
        print("=" * 60)
        posthoc_test = result.get('posthoc_test', 'nemenyi')
        print(f"Method: {posthoc_test}")
        for row in result['post_hoc_results']:
            status = "SIGNIFICANT" if row.get('Significant') else "Not significant"
            if 'P-value' in row:
                print(f"{row['Model 1']} vs {row['Model 2']}: p = {row['P-value']:.6f} ({status})")
            elif 'Rank Difference' in row and 'Critical Distance' in row:
                diff = row['Rank Difference']
                cd = row['Critical Distance']
                print(f"{row['Model 1']} vs {row['Model 2']}: Rank diff = {diff:.3f}, CD = {cd:.3f} ({status})")

    # Bayesian results
    if result.get('bayesian_results'):
        print("\nBAYESIAN ANALYSIS RESULTS:")
        print("=" * 60)
        for comparison in result['bayesian_results']:
            m1 = comparison['Model 1']
            m2 = comparison['Model 2']
            pl = comparison.get('P_left(M1>M2)', 0.0)
            pe = comparison.get('P_rope(|diff|<=ROPE)', 0.0)
            pr = comparison.get('P_right(M2>M1)', 0.0)
            decision = comparison.get('Decision', '—')
            print(f"{m1} vs {m2}: P_left={pl:.3f}, P_rope={pe:.3f}, P_right={pr:.3f} | Decision: {decision}")


def save_results(result: Dict[str, Any], output_path: str, models: list, analysis_type: str, metric: str) -> None:
    """Save results to JSON file."""
    os.makedirs(output_path, exist_ok=True)

    # Use simplified filename for the statistical results
    output_file = os.path.join(output_path, f"statistical_results_{metric}.json")

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"\nResults saved to: {output_file}")


def print_data_summary(results: Dict[str, list], models: list) -> None:
    """Print summary of loaded data."""
    print("Loaded data summary:")
    for model in models:
        scores = results.get(model, [])
        if len(scores) > 0:
            import numpy as np
            print(f"  {model}: {len(scores)} values, mean = {np.mean(scores):.4f}")
        else:
            print(f"  {model}: No data found")
    print()
