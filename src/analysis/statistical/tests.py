import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
import baycomp
from itertools import combinations
from typing import List, Tuple, Dict, Any, Optional


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


def friedman_test(data_matrix: List[np.ndarray]) -> Tuple[float, float]:
    """Perform Friedman test."""
    statistic, p_value = friedmanchisquare(*data_matrix)
    return float(statistic), float(p_value)


def kendalls_w(data_matrix: List[np.ndarray]) -> float:
    """Calculate Kendall's W effect size."""
    n_models = len(data_matrix)
    n_observations = len(data_matrix[0])

    data = np.array(data_matrix).T
    ranks = np.array([rankdata(row) for row in data])
    rank_sums = np.sum(ranks, axis=0)

    mean_rank_sum = np.mean(rank_sums)
    sum_squared_deviations = np.sum((rank_sums - mean_rank_sum) ** 2)

    return (12 * sum_squared_deviations) / (n_observations ** 2 * (n_models ** 3 - n_models))


def pairwise_comparisons(data_matrix: List[np.ndarray], models: List[str],
                        alpha: float = 0.05) -> pd.DataFrame:
    """Perform pairwise Wilcoxon tests with Bonferroni correction."""
    n_comparisons = len(models) * (len(models) - 1) // 2
    corrected_alpha = alpha / n_comparisons
    results = []

    for i, j in combinations(range(len(models)), 2):
        try:
            statistic, p_value = wilcoxon(data_matrix[i], data_matrix[j])
            results.append({
                'Model 1': models[i],
                'Model 2': models[j],
                'Statistic': float(statistic),
                'P-value': float(p_value),
                'Corrected Alpha': corrected_alpha,
                'Significant': p_value < corrected_alpha
            })
        except ValueError as e:
            print(f"Warning: Wilcoxon test failed for {models[i]} vs {models[j]}: {e}")

    return pd.DataFrame(results)


def pairwise_bayesian_signed_rank(
    data_matrix: List[np.ndarray],
    models: List[str],
    rope: float = 0.01,
    nsamples: int = 20000,
    prior: float = 0.5,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Perform pairwise Bayesian signed-rank tests across all model pairs.

    Uses baycomp.SignedRankTest (two_on_multiple with 1D inputs).
    Returns probabilities:
      - P_left: Prob(model1 > model2 by more than ROPE)
      - P_rope: Prob(models are practically equivalent within ROPE)
      - P_right: Prob(model2 > model1 by more than ROPE)
    """
    results = []
    for i, j in combinations(range(len(models)), 2):
        x, y = np.asarray(data_matrix[i]), np.asarray(data_matrix[j])

        # baycomp.two_on_multiple returns (p_left, p_rope, p_right)
        p_left, p_rope, p_right = baycomp.two_on_multiple(
            x, y, rope=rope, nsamples=nsamples, prior=prior, random_state=random_state
        )

        # Decide a 'winner' if one side has higher probability than the other and ROPE
        # probs = {'left': p_left, 'rope': p_rope, 'right': p_right}
        winner = None
        if max(p_left, p_right) > p_rope:
            winner = models[i] if p_left > p_right else models[j]

        results.append({
            'Model 1': models[i],
            'Model 2': models[j],
            'ROPE': rope,
            'P_left(M1>M2)': float(p_left),
            'P_rope(|diff|<=ROPE)': float(p_rope),
            'P_right(M2>M1)': float(p_right),
            'Winner': winner
        })

    return pd.DataFrame(results)


def create_result_summary(statistic: float, p_value: float, kendalls_w_value: float,
                         models: List[str], data_matrix: List[np.ndarray],
                         post_hoc_df: pd.DataFrame, alpha: float,
                         analysis_type: str, metric: str,
                         bayesian_post_hoc_df: Optional[pd.DataFrame] = None,
                         bayesian_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create structured result summary."""
    return {
        'analysis_type': analysis_type,
        'friedman_statistic': statistic,
        'p_value': p_value,
        'kendalls_w': kendalls_w_value,
        'significant': p_value < alpha,
        'models': models,
        'metric': metric,
        'alpha': alpha,
        'model_performance': {
            model: {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'values': scores.tolist()
            }
            for model, scores in zip(models, data_matrix)
        },
        'post_hoc_results': post_hoc_df.to_dict('records') if not post_hoc_df.empty else [],
        'bayesian_post_hoc_results': (
            bayesian_post_hoc_df.to_dict('records') if (bayesian_post_hoc_df is not None and not bayesian_post_hoc_df.empty) else []
        ),
        'bayesian_params': bayesian_params or {}
    }
