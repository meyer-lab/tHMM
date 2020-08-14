""" This file plots the AIC for the real data. """

import numpy as np

from ..Analyze import run_Analyze_AIC
from ..data.Lineage_collections import Gemcitabine_Control

# States to evaluate with the model
from ..states.StateDistributionGaPhs import StateDistribution
from .figureCommon import getSetup, subplotLabel
from .figureS11 import run_AIC, figure_maker


desired_num_states = np.arange(1, 8)


def makeFigure():
    """
    Makes figure 10.
    """
    ax, f = getSetup((13.333, 3.333), (1, 4))
    desired_num_states = np.arange(1, 8)

    # making lineages and finding AICs (assign number of lineages here)
    AIC = run_AIC(Gemcitabine_Control)

    # Finding proper ylim range for all 4 censored graphs and rounding up
    upper_ylim_censored = int(1 + max(np.max(np.ptp(AIC[0], axis=0)), np.max(np.ptp(
        AIC[1], axis=0)), np.max(np.ptp(AIC[2], axis=0)), np.max(np.ptp(AIC[3], axis=0))) / 25.0) * 25

    upper_ylim = [upper_ylim_censored]

    # Plotting AICs
    for idx, a in enumerate(AIC):
        figure_maker(ax[idx], a, (idx % 4) + 1,
                     upper_ylim[int(idx / 4)], idx > 3)
    subplotLabel(ax)

    return f


def run_AIC(lineages):
    """
    Run AIC for experimental data.
    """

    # Storing AICs into array
    AICs = np.empty((len(desired_num_states), len(lineages)))
    output = run_Analyze_AIC(lineages, desired_num_states)
    for idx in range(len(desired_num_states)):
        AIC, _ = output[idx][0].get_AIC(output[idx][2])
        AICs[idx] = np.array([ind_AIC for ind_AIC in AIC])

    return AICs
