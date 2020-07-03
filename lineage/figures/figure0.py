""" This file only shows the binary tree plot of the lineage. """

from .figureCommon import getSetup

def makeFigure():
    """
    Makes fig 0.
    """

    # Get list of axis objects
    ax, f = getSetup((4.0, 4.0), (1, 1))

    figureMaker(ax)

    return f

def figureMaker(ax):
    """
    This makes figure 3B.
    """
    # cartoon to show different shapes --> similar shapes
    ax[0].axis('off')
