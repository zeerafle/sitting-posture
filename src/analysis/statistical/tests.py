import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from itertools import combinations
import autorank  # New import

def prepare_data_matrix(results: Dict[str, List[float]], models: List[str]) -> List[np.ndarray]:
    """Prepare data matrix for statistical tests."""
    data_matrix = []
    for model in models:
        scores = results.get(model, [])
        valid_scores = [score for score in scores if score is not None]
        data_matrix.append(valid_scores)

    # Handle inconsistent lengths
    lengths = [len(scores) for scores in data_matrix]
    if len(set(lengths)) > 1:
        min_length = min(lengths)
        print(f"Warning: Models have different numbers of results. Using first {min_length} results.")
        data_matrix = [scores[:min_length] for scores in data_matrix]

    # Validate data
    if len(data_matrix) < 2:
        raise ValueError("Need at least 2 models to perform statistical tests")
    if any(len(scores) < 2 for scores in data_matrix):
        raise ValueError("Need at least 2 observations per model")

    return [np.array(scores, dtype=float) for scores in data_matrix]

def convert_to_dataframe(data_matrix: List[np.ndarray], models: List[str]) -> pd.DataFrame:
    """Convert data matrix to a pandas DataFrame for autorank."""
    # Convert matrix to DataFrame in wide format
    n_samples = len(data_matrix[0])
    df = pd.DataFrame()
    for i, model in enumerate(models):
        df[model] = data_matrix[i][:n_samples]
    return df

def run_autorank_analysis(data_matrix: List[np.ndarray], models: List[str],
                         alpha: float = 0.05,
                         approach: str = 'frequentist',
                         verbose: bool = False,
                         rope: Optional[float] = None) -> Dict[str, Any]:
    """Run autorank analysis on the data matrix."""
    # Convert to DataFrame in the format expected by autorank
    df = convert_to_dataframe(data_matrix, models)

    # Run autorank with the appropriate parameters
    if approach == 'bayesian':
        # For Bayesian approach, we need a ROPE value
        # If not specified, use a default of 0.01 (1%)
        rope_value = 0.01 if rope is None else rope
        result = autorank.autorank(df, alpha=alpha, approach=approach, verbose=verbose, rope=rope_value)
    else:
        # For frequentist approach, don't pass the rope parameter
        result = autorank.autorank(df, alpha=alpha, approach=approach, verbose=verbose)

    # Return the result for further processing
    return result

def create_result_summary(autorank_result: Any,
                         models: List[str], data_matrix: List[np.ndarray],
                         alpha: float, analysis_type: str, metric: str,
                         higher_is_better: bool = True,
                         cd_diagram_file: Optional[str] = None) -> Dict[str, Any]:
    """Create structured result summary from autorank result."""
    # Extract basic statistics
    rankdf = autorank_result.rankdf

    # Extract p-value and test information
    p_value = autorank_result.pvalue if hasattr(autorank_result, 'pvalue') else None
    omnibus_test = autorank_result.omnibus if hasattr(autorank_result, 'omnibus') else None
    posthoc_test = autorank_result.posthoc if hasattr(autorank_result, 'posthoc') else None
    cd_value = autorank_result.cd if hasattr(autorank_result, 'cd') else None

    # Create performance summary
    model_performance = {}
    for model, scores in zip(models, data_matrix):
        model_performance[model] = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'median': float(np.median(scores)),
            'mad': float(np.median(np.abs(scores - np.median(scores)))),
            'values': scores.tolist()
        }

    # Extract post-hoc results if available (for frequentist approach)
    post_hoc_results = []
    if hasattr(autorank_result, 'rankdf') and 'meanrank' in autorank_result.rankdf.columns:
        # Extract meanrank pairs for all model combinations
        for i, j in combinations(range(len(models)), 2):
            model1, model2 = models[i], models[j]
            rank1 = float(rankdf.loc[model1, 'meanrank']) if model1 in rankdf.index else float('nan')
            rank2 = float(rankdf.loc[model2, 'meanrank']) if model2 in rankdf.index else float('nan')
            rank_diff = abs(rank1 - rank2)
            significant = rank_diff > cd_value if cd_value is not None else False

            post_hoc_results.append({
                'Model 1': model1,
                'Model 2': model2,
                'Rank Difference': rank_diff,
                'Critical Distance': cd_value,
                'Significant': significant
            })

    # Extract Bayesian results if available
    bayesian_results = []
    if hasattr(autorank_result, 'posterior_matrix') and autorank_result.posterior_matrix is not None:
        posterior_matrix = autorank_result.posterior_matrix
        decision_matrix = autorank_result.decision_matrix

        for i, j in combinations(range(len(models)), 2):
            model1, model2 = models[i], models[j]
            if model1 in posterior_matrix.index and model2 in posterior_matrix.columns:
                posterior = posterior_matrix.loc[model1, model2]
                decision = decision_matrix.loc[model1, model2] if decision_matrix is not None else 'unknown'

                if posterior is not None and posterior != 'NaN':
                    # Parse posterior tuple (p_left, p_rope, p_right)
                    try:
                        p_vals = eval(posterior) if isinstance(posterior, str) else posterior
                        p_left, p_rope, p_right = p_vals

                        bayesian_results.append({
                            'Model 1': model1,
                            'Model 2': model2,
                            'P_left(M1>M2)': float(p_left),
                            'P_rope(|diff|<=ROPE)': float(p_rope),
                            'P_right(M2>M1)': float(p_right),
                            'Decision': decision
                        })
                    except (ValueError, TypeError):
                        # Skip if posterior values can't be parsed
                        pass

    # Create final result summary
    result = {
        'analysis_type': analysis_type,
        'p_value': p_value,
        'significant': p_value < alpha if p_value is not None else None,
        'models': models,
        'metric': metric,
        'alpha': alpha,
        'model_performance': model_performance,
        'post_hoc_results': post_hoc_results,
        'bayesian_results': bayesian_results,
        'omnibus_test': omnibus_test,
        'posthoc_test': posthoc_test,
        'higher_is_better': higher_is_better
    }

    # Add CD value if available
    if cd_value is not None:
        result['cd_value'] = float(cd_value)

    # Add CD diagram path if provided
    if cd_diagram_file is not None:
        result['cd_diagram_file'] = cd_diagram_file

    # Add average ranks if available
    if hasattr(autorank_result, 'rankdf') and 'meanrank' in autorank_result.rankdf.columns:
        result['average_ranks'] = {model: float(rankdf.loc[model, 'meanrank'])
                                 if model in rankdf.index else float('nan')
                                 for model in models}

    return result
