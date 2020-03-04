"""
Contains utilities and functions that are commonly used or shared amongst
the figure creation files.
"""

from string import ascii_lowercase
import numpy as np
from matplotlib import gridspec, pyplot as plt


def getSetup(figsize, gridd):
    """Setup figures."""

    # Setup plotting space
    f = plt.figure(figsize=figsize)

    # Make grid
    gs1 = gridspec.GridSpec(*gridd)

    # Get list of axis objects
    ax = [f.add_subplot(gs1[x]) for x in range(gridd[0] * gridd[1])]

    return (ax, f)


def moving_average(a, n=50):
    """
    Calculates the moving average.
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def subplotLabel(axs):
    """Sublot labels"""
    for ii, ax in enumerate(axs):
        ax.text(-0.2, 1.25, ascii_lowercase[ii], transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")
