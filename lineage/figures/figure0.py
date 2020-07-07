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

    lineage_uncensored = LineageTree.init_from_parameters(pi, T, E2, desired_num_cells=2**8 - 1)
    plotLineage(lineage_uncensored, 'lineage/figures/cartoons/lineage_uncensored.svg')

    lineage_censored = LineageTree.init_from_parameters(pi, T, E2, desired_num_cells=2**8 - 1, censor_condition=3, desired_experiment_time=200)
    plotLineage(lineage_censored, 'lineage/figures/cartoons/lineage_censored.svg')

    return f


def figureMaker(ax):
    """
    This makes figure 0.
    """
    ax[0].axis('off')
