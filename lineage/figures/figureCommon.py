import seaborn as sns
import numpy as np
from matplotlib import gridspec, pyplot as plt


def getSetup(figsize, gridd):
    """Setup figures."""

    plt.rc('font', **{'family': 'sans-serif', 'size': 25})
    # for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})
    plt.rc('text', usetex=True)
    plt.rc('xtick', **{'labelsize': 'medium'})
    plt.rc('ytick', **{'labelsize': 'medium'})

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


def subplotLabel(ax, letter, hstretch=1):
    """Sublot labels"""
    ax.text(-0.2 / hstretch, 1.2, letter, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top')
