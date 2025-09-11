"""Data loading utilities for statistical analysis."""
import os
import json
from typing import Dict, List


def load_json_data(file_path: str) -> Dict:
    """Load data from JSON file with error handling."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {file_path}: {e}")
        return {}


def extract_metric_values(data: Dict, metric: str) -> List[float]:
    """Extract metric values from loaded data."""
    if metric in data and isinstance(data[metric], list):
        return [float(x) for x in data[metric] if x is not None]
    return []


def load_cv_results(dvclive_path: str, model_name: str, metric: str) -> List[float]:
    """Load CV results for a model."""
    all_scores = []
    for view in ['front', 'left', 'right']:
        cv_path = os.path.join(dvclive_path, model_name, view, "cv_results.json")
        data = load_json_data(cv_path)
        # Try different possible keys for the metric
        for key in [f'test_{metric}', f'cv_{metric}_scores', f'{metric}_scores', 'test_score']:
            if key in data and isinstance(data[key], list):
                all_scores.extend([float(x) for x in data[key] if x is not None])
                break
    return all_scores


def load_combined_results(dvclive_path: str, model_name: str, metric: str) -> List[float]:
    """Load combined CV results for a model."""
    cv_path = os.path.join(dvclive_path, model_name, "combined", "cv_results.json")
    data = load_json_data(cv_path)
    for key in [f'test_{metric}', f'cv_{metric}_scores', f'{metric}_scores', 'test_score']:
        if key in data and isinstance(data[key], list):
            return [float(x) for x in data[key] if x is not None]
    return []


def load_loso_results(dvclive_path: str, model_name: str, metric: str) -> List[float]:
    """Load LOSO results for a model.

    Checks for metrics in both fold_metrics.json and metrics.json in the loso directory.
    """
    # Try fold_metrics.json first
    fold_metrics_path = os.path.join(dvclive_path, model_name, "loso", "fold_metrics.json")
    data = load_json_data(fold_metrics_path)

    if metric in data:
        return extract_metric_values(data, metric)

    # If not found, try metrics.json
    metrics_path = os.path.join(dvclive_path, model_name, "loso", "metrics.json")
    data = load_json_data(metrics_path)

    # Extract metric values from individual folds in metrics.json
    values = []
    for fold_key in [k for k in data.keys() if k.startswith('fold_')]:
        if metric in data[fold_key]:
            values.append(float(data[fold_key][metric]))

    return values


def load_ablation_results(dvclive_path: str, model_configs: List[str], metric: str) -> Dict[str, List[float]]:
    """Load ablation study results for multiple model configurations."""
    results = {}
    for config in model_configs:
        loso_path = os.path.join(dvclive_path, config, "loso_metrics.json")
        data = load_json_data(loso_path)
        results[config] = extract_metric_values(data, metric)
    return results


def load_model_results(dvclive_path: str, models: List[str], metric: str, analysis_type: str) -> Dict[str, List[float]]:
    """Load results for all models based on analysis type."""
    results = {}

    for model in models:
        if analysis_type == 'individual':
            results[model] = load_cv_results(dvclive_path, model, metric)
        elif analysis_type == 'combined':
            results[model] = load_combined_results(dvclive_path, model, metric)
        elif analysis_type == 'loso':
            results[model] = load_loso_results(dvclive_path, model, metric)
        elif analysis_type == 'ablation':
            # For ablation, models are actually config names
            results = load_ablation_results(dvclive_path, models, metric)
            break

    return results
