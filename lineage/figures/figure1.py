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
    ax, f = getSetup((7, 7), (1, 1))

    subplotLabel(ax)

    return f
