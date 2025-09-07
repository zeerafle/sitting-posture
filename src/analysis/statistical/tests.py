import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
import baycomp
from itertools import combinations
from typing import List, Tuple, Dict, Any, Optional

# Optional deps for post-hoc and p-value adjustments
try:
    import scikit_posthocs as sp
except Exception:  # pragma: no cover
    sp = None

try:
    from statsmodels.stats.multitest import multipletests
except Exception:  # pragma: no cover
    multipletests = None


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


def average_ranks(data_matrix: List[np.ndarray], higher_is_better: bool = True) -> np.ndarray:
    """
    Compute average ranks across datasets/folds.
    For metrics where higher is better (e.g., accuracy), we invert sign so that rank 1 is best.
    """
    # data shape: models x datasets
    mat = np.vstack(data_matrix)
    if higher_is_better:
        mat = -mat  # invert so "best" (largest) gets smallest rank

    # rank per dataset (column)
    ranks = np.array([rankdata(col, method='average') for col in mat.T]).T  # models x datasets
    return ranks.mean(axis=1)


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


def _matrix_to_pair_list(pmat: pd.DataFrame, labels: List[str], alpha: float,
                         method_name: str) -> pd.DataFrame:
    """Convert a symmetric p-value matrix to a long pairwise list with significance."""
    # Ensure index/columns match labels order
    if list(pmat.index) != labels or list(pmat.columns) != labels:
        pmat = pmat.copy()
        pmat.index = labels
        pmat.columns = labels

    rows = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            p = float(pmat.iloc[i, j])
            rows.append({
                'Model 1': labels[i],
                'Model 2': labels[j],
                'P-value': p,
                'Alpha': alpha,
                'Method': method_name,
                'Significant': p < alpha
            })
    return pd.DataFrame(rows)


def _rename_pmat(pmat: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    # Some posthocs return numeric labels 0..k-1; enforce model names order
    pmat = pmat.copy()
    if pmat.shape[0] == len(models):
        pmat.index = models
    if pmat.shape[1] == len(models):
        pmat.columns = models
    return pmat


def posthoc_after_friedman(
    data_matrix: List[np.ndarray],
    models: List[str],
    alpha: float = 0.05,
    method: str = 'nemenyi'
) -> Dict[str, Any]:
    """
    Post-hoc procedures for k >= 3 after significant Friedman:
      - 'nemenyi': Nemenyi for unreplicated blocks (preferred default)
      - 'holm': Conover-Friedman (or pairwise Wilcoxon) with Holm correction
      - 'shaffer': Try to use scikit-posthocs if available; otherwise fallback to Holm
    Returns dict: {'method': str, 'pmatrix': DataFrame, 'pairs': DataFrame}
    """
    k = len(models)
    if k < 3:
        return {'method': 'none', 'pmatrix': pd.DataFrame(), 'pairs': pd.DataFrame()}

    mat = np.vstack(data_matrix).T  # blocks x groups

    # Helper to return formatted result
    def pack(pmat: pd.DataFrame, used_method: str) -> Dict[str, Any]:
        pmat2 = _rename_pmat(pmat, models)
        pairs = _matrix_to_pair_list(pmat2, models, alpha, used_method)
        return {'method': used_method, 'pmatrix': pmat2, 'pairs': pairs}

    method = (method or 'nemenyi').lower()

    if method == 'nemenyi':
        if sp is None:
            raise ImportError("scikit-posthocs is required for Nemenyi post-hoc. pip install scikit-posthocs")
        pmat = sp.posthoc_nemenyi_friedman(mat)
        return pack(pmat, 'nemenyi')

    if method == 'holm':
        if sp is not None:
            # Conover-Friedman supports p_adjust via statsmodels backend; if not, we’ll do manual Holm
            try:
                pmat = sp.posthoc_conover_friedman(mat, p_adjust='holm')
                return pack(pmat, 'conover-holm')
            except Exception:
                pass

        # Fallback: pairwise Wilcoxon (signed-rank across blocks) with Holm correction via statsmodels
        if multipletests is None:
            raise ImportError("statsmodels is required for Holm correction fallback. pip install statsmodels")
        pairs = []
        for i, j in combinations(range(k), 2):
            stat, p = wilcoxon(data_matrix[i], data_matrix[j])
            pairs.append((i, j, p))
        raw_p = [p for (_, _, p) in pairs]
        rej, p_adj, _, _ = multipletests(raw_p, method='holm', alpha=alpha)
        # Build p-matrix
        pmat = pd.DataFrame(np.ones((k, k)), index=models, columns=models, dtype=float)
        for (idx, (i, j, _)) in enumerate(pairs):
            pmat.iloc[i, j] = p_adj[idx]
            pmat.iloc[j, i] = p_adj[idx]
        np.fill_diagonal(pmat.values, 1.0)
        return pack(pmat, 'wilcoxon-holm')

    if method == 'shaffer':
        # scikit-posthocs does not expose Shaffer in a stable public API (to my knowledge).
        # We fallback to Holm and label accordingly.
        try:
            return posthoc_after_friedman(data_matrix, models, alpha, method='holm') | {'method': 'shaffer(fallback-holm)'}
        except Exception as e:
            raise e

    # Default to Nemenyi
    return posthoc_after_friedman(data_matrix, models, alpha, method='nemenyi')


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
                         bayesian_params: Optional[Dict[str, Any]] = None,
                         post_hoc_matrix: Optional[pd.DataFrame] = None,
                         post_hoc_method: Optional[str] = None,
                         avg_ranks: Optional[List[float]] = None,
                         cd_value: Optional[float] = None,
                         cd_diagram_file: Optional[str] = None) -> Dict[str, Any]:
    """Create structured result summary."""
    result = {
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
        'post_hoc_results': post_hoc_df.to_dict('records') if (post_hoc_df is not None and not post_hoc_df.empty) else [],
        'bayesian_post_hoc_results': (
            bayesian_post_hoc_df.to_dict('records') if (bayesian_post_hoc_df is not None and not bayesian_post_hoc_df.empty) else []
        ),
        'bayesian_params': bayesian_params or {}
    }

    if post_hoc_matrix is not None and not post_hoc_matrix.empty:
        result['post_hoc_method'] = post_hoc_method or ''
        # Store matrix as nested dict
        result['post_hoc_matrix'] = post_hoc_matrix.round(6).to_dict()

    if avg_ranks is not None:
        result['average_ranks'] = {m: float(r) for m, r in zip(models, avg_ranks)}
    if cd_value is not None:
        result['cd_value'] = float(cd_value)
    if cd_diagram_file is not None:
        result['cd_diagram_file'] = cd_diagram_file

    return result
