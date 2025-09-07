"""
CD diagram utilities.

Tries Orange's compute_CD/graph_ranks first. If unavailable, falls back to:
 - computing the studentized-range quantile using statsmodels.libqsturng.qsturng
 - computing CD = q * sqrt(k*(k+1)/(6*N))
 - plotting a simple Critical Difference diagram using matplotlib
"""
from typing import List, Optional
import math

def _try_orange():
    try:
        from Orange.evaluation import compute_CD, graph_ranks  # type: ignore
        return compute_CD, graph_ranks
    except Exception:
        return None, None


def _compute_cd_with_statsmodels(k: int, n_datasets: int, alpha: float):
    """
    Compute critical difference using statsmodels' qsturng (studentized range).
    CD = q_alpha * sqrt(k*(k+1)/(6*N))

    statsmodels.stats.libqsturng.qsturng(prob, k, df) returns the quantile for
    the studentized range distribution. For Nemenyi we usually use df = inf,
    approximated with a large number or np.inf if supported.
    """
    try:
        from statsmodels.stats.libqsturng import qsturng  # type: ignore
    except Exception as e:
        raise ImportError(
            "Neither Orange.compute_CD nor statsmodels.qsturng are available. "
            "Install Orange3 (preferred) or statsmodels+matplotlib to draw CD diagrams."
        ) from e

    # qsturng takes 'prob' (e.g., 1-alpha), 'r' (number of groups), and 'v' (df)
    # For unreplicated comparisons (Nemenyi), df -> inf; statsmodels accepts np.inf
    prob = 1.0 - alpha
    # Some versions require a finite df; try np.inf and fallback to a large df
    try:
        q = qsturng(prob, k, float("inf"))
    except Exception:
        # fallback to a very large df
        q = qsturng(prob, k, 1e8)
    cd = q * math.sqrt((k * (k + 1)) / (6.0 * n_datasets))
    return float(cd)


def _plot_cd_matplotlib(avg_ranks: List[float], model_names: List[str], cd: float, outfile: Optional[str] = None):
    """Simple CD diagram plotting using matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise ImportError("matplotlib is required for plotting CD diagram fallback. pip install matplotlib") from e

    k = len(model_names)
    # Sort models by rank (low = better)
    sorted_idx = sorted(range(k), key=lambda i: avg_ranks[i])
    ranks_sorted = [avg_ranks[i] for i in sorted_idx]
    names_sorted = [model_names[i] for i in sorted_idx]

    # figure sizing
    width = max(6, k * 0.6)
    height = 2.4 + (0.0)
    fig, ax = plt.subplots(figsize=(width, height))

    # x-range
    min_rank = min(ranks_sorted) - 0.5
    max_rank = max(ranks_sorted) + 0.5
    ax.set_xlim(min_rank, max_rank)
    ax.set_ylim(0, 2.5)

    # draw baseline
    ax.hlines(1.0, min_rank, max_rank, color="black", linewidth=0.8)

    # plot points and labels
    y = 1.0
    for x, name in zip(ranks_sorted, names_sorted):
        ax.plot(x, y, 'o', color='black')
        ax.text(x, y - 0.18, name, rotation=45, ha='right', va='top', fontsize=9)

    # draw CD bar at top-left
    cd_x = max_rank - cd
    # place CD annotation on top-right area
    top_y = 1.8
    ax.hlines(top_y, cd_x, cd_x + cd, color='black', linewidth=2)
    mid = cd_x + cd / 2.0
    ax.text(mid, top_y + 0.05, f"CD = {cd:.3f}", ha='center', va='bottom', fontsize=9)

    # Find contiguous groups where rank difference <= cd
    groups = []
    current_group = [0]
    for i in range(1, k):
        if ranks_sorted[i] - ranks_sorted[i - 1] <= cd + 1e-12:
            current_group.append(i)
        else:
            if len(current_group) > 1:
                groups.append(list(current_group))
            current_group = [i]
    if len(current_group) > 1:
        groups.append(list(current_group))

    # For each group draw a bracket above the baseline
    bracket_y = 1.35
    bracket_height = 0.08
    for group in groups:
        x_start = ranks_sorted[group[0]]
        x_end = ranks_sorted[group[-1]]
        ax.hlines(bracket_y, x_start, x_end, color='black', linewidth=1.5)
        # vertical ticks
        ax.vlines([x_start, x_end], bracket_y - bracket_height, bracket_y + bracket_height, color='black', linewidth=1.2)

    ax.set_yticks([])
    ax.set_xlabel("Average rank (lower = better)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, bbox_inches='tight')
        plt.close(fig)
    return fig


def draw_cd_diagram(avg_ranks: List[float], model_names: List[str], n_datasets: int,
                    alpha: float = 0.05, test: str = 'nemenyi',
                    outfile: Optional[str] = None) -> float:
    """
    Draw a Critical Difference (CD) diagram and save to outfile.

    Behavior:
      1. Try Orange.evaluation.compute_CD & graph_ranks
      2. Fallback: compute q using statsmodels.qsturng and draw with matplotlib

    Returns:
      cd (float)
    """
    # Try Orange first
    compute_CD, graph_ranks = _try_orange()
    k = len(model_names)

    if compute_CD and graph_ranks:
        # If Orange's compute_CD exists, use it (keeps the exact look)
        try:
            cd = compute_CD(avg_ranks, n_datasets, alpha=alpha, test=test)  # type: ignore
            fig = graph_ranks(avg_ranks, model_names, cd=cd, width=6, textspace=1.5)  # type: ignore
            # Save
            if outfile:
                try:
                    import matplotlib.pyplot as plt
                    fig.savefig(outfile, bbox_inches='tight')
                    plt.close(fig)
                except Exception:
                    fig.savefig(outfile)
            return float(cd)
        except Exception as e:
            # fall through to fallback implementation
            pass

    # Fallback: compute CD via statsmodels
    cd = _compute_cd_with_statsmodels(k, n_datasets, alpha)
    # Plot with matplotlib fallback
    _plot_cd_matplotlib(avg_ranks, model_names, cd, outfile=outfile)
    return float(cd)
