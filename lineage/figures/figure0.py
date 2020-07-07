""" This file only shows the binary tree plot of the lineage. """

from .figureCommon import (
    getSetup,
    pi,
    T,
    E2
)
from ..LineageTree import LineageTree
from ..plotTree import plotLineage


def makeFigure():
    """
    Makes fig 0.
    """

    # Get list of axis objects
    ax, f = getSetup((6.0, 3.0), (1, 1))

    figureMaker(ax)

    return f


def figureMaker(ax):
    """
    This makes figure 0.
    """
    ax[0].axis('off')
