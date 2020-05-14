"""
File: figure1.py
Purpose: Generates figure 1.

Figure 1 is the distribution of cells in a state over generations (uncensored) and over time.
"""

from .figureCommon import getSetup, subplotLabel


def makeFigure():
    """
    Makes figure 1.
    """

    # Get list of axis objects
    ax, f = getSetup((7, 7), (2, 2))

    figureMaker(ax)

    subplotLabel(ax)

    return f


def figureMaker(ax):
    """
    Creates the data for figure 1.
    """
