""" This file contains functions for plotting different phenotypes in the manuscript. """

import pandas as pd
import seaborn as sns
from .figureCommon import (
    getSetup,
    subplotLabel,
    pi,
    T,
    E2
)
from ..LineageTree import LineageTree
from ..Analyze import Analyze, Results
from .figure2A import forHistObs, figureMaker2


def makeFigure():
    """
    Makes fig 2.
    """

    # Get list of axis objects
    ax, f = getSetup((10.0, 7.5), (3, 4))
    X = [LineageTree.init_from_parameters(pi, T, E2, desired_num_cells=2**8 - 1, censor_condition=3, desired_experiment_time=500)]
    results_dict = Results(*Analyze(X, 2))
    pred_states_by_lineage = results_dict["switched_pred_states_by_lineage"][0]  # only one lineage

    figureMaker2(ax, forHistObs(X, pred_states_by_lineage))

    subplotLabel(ax)

    return f
