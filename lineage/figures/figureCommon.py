"""
Contains utilities and functions that are commonly used or shared amongst
the figure creation files.
"""

from string import ascii_lowercase
import numpy as np
from matplotlib import gridspec, pyplot as plt
import seaborn as sns

from ..StateDistribution import StateDistribution

# pi: the initial probability vector
pi = np.array([0.6, 0.4], dtype="float")

# T: transition probability matrix
T = np.array([[0.85, 0.15],
              [0.15, 0.85]], dtype="float")

# bern, gamma_a, gamma_scale
E = [StateDistribution(0.99, 20, 5), StateDistribution(0.88, 10, 1)]


def getSetup(figsize, gridd):
    """ Establish figure set-up with subplots. """
    sns.set(style="whitegrid", font_scale=0.7, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Setup plotting space and grid
    f = plt.figure(figsize=figsize, constrained_layout=True)
    gs1 = gridspec.GridSpec(*gridd, figure=f)

    # Get list of axis objects
    ax = list()
    for x in range(gridd[0] * gridd[1]):
        ax.append(f.add_subplot(gs1[x]))

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
