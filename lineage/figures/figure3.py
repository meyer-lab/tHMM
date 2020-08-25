"""
File: figure1.py
Purpose: Generates figure 1.
Figure 1 is the tHMM model and its purpose.
"""

from .figureCommon import getSetup


def makeFigure():
    """
    Makes figure 3.
    """
    # Get list of axis objects
    ax, f = getSetup((8.0, 3.0), (1, 1))
    figureMaker(ax)

    return f


def figureMaker(ax):
    """
    Makes figure 3.
    """
    i = 0
    ax[i].axis('off')
