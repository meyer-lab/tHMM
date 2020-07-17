"""
File: figure1.py
Purpose: Generates figure 1.
Figure 1 is the tHMM model and its purpose.
"""

from .figureCommon import getSetup, subplotLabel


def makeFigure():
    """
    Makes figure 1.
    """

    # Get list of axis objects
    ax, f = getSetup((7, 10 / 3), (1, 1))
    figureMaker(ax)
    subplotLabel(ax)

    return f


def figureMaker(ax):
    """
    Makes figure 1.
    """
    i = 0
    ax[i].axis('off')
