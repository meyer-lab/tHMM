"""
File: figure0.py
Purpose: Generates figure 0.

Figure 0 is the distribution of cells in a state over generations (uncensored) and over time.
"""
from matplotlib.ticker import MaxNLocator

from .figureCommon import getSetup, pi, T, E, subplotLabel
from ..LineageTree import LineageTree

def makeFigure():
    """
    Makes figure 2.
    """

    # Get list of axis objects
    ax, f = getSetup((7, 7), (2, 2))

    subplotLabel(ax)

    return f
   