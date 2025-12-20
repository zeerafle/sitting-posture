"""
CD diagram utilities using autorank.
"""
from typing import List, Optional
from loguru import logger
import matplotlib.pyplot as plt
import autorank

def draw_cd_diagram(autorank_result: Any, models: List[str], outfile: Optional[str] = None) -> float:
    """
    Draw a Critical Difference (CD) diagram using autorank and save to outfile.

    Returns:
      cd (float)
    """
    try:
        # Use autorank's plot_stats function to create CD diagram
        fig = autorank.plot_stats(autorank_result)

        # Save the plot if outfile is specified
        if outfile:
            fig.savefig(outfile, bbox_inches='tight')
            plt.close(fig)

        # Return the CD value
        return float(autorank_result.cd) if hasattr(autorank_result, 'cd') else 0.0

    except Exception as e:
        logger.error(f"Error using autorank to create CD diagram: {e}")
        raise ValueError("Failed to create CD diagram using autorank")
