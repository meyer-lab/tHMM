"""
File: figure10.py
Purpose: Generates figure 10.

AIC.
"""
import numpy as np


from ..Analyze import run_Analyze_AIC
from ..LineageTree import LineageTree

# States to evaluate with the model
from ..states.StateDistributionGamma import StateDistribution

from .figureCommon import getSetup, lineage_good_to_analyze, subplotLabel
from .figureS10 import run_AIC, figure_maker


desired_num_states = np.arange(1, 8)


def makeFigure():
    """
    Makes figure 10.
    """
    ax, f = getSetup((10, 6), (2, 4))
    desired_num_states = np.arange(1, 8)

    # Setting up state distributions and E
    Sone = StateDistribution(0.99, 0.8, 8, 7, 5, 8)
    Stwo = StateDistribution(0.9, 0.75, 6, 4, 12, 10)
    Sthree = StateDistribution(0.85, 0.7, 10, 7, 6, 2)
    Sfour = StateDistribution(0.8, 0.9, 12, 9, 3, 6)
    Eone = [Sone, Sone]
    Etwo = [Sone, Stwo]
    Ethree = [Sone, Stwo, Sthree]
    Efour = [Sone, Stwo, Sthree, Sfour]
    E = [Eone, Etwo, Ethree, Efour, Eone, Etwo, Ethree, Efour]

    # making lineages and finding AICs (assign number of lineages here)
    AIC = [run_AIC(.1, e, 10, idx > 4) for idx, e in enumerate(E)]

    # Finding proper ylim range for all 4 uncensored graphs and rounding up
    upper_ylim_uncensored = int(1 + max(np.max(np.ptp(AIC[0], axis=0)), np.max(np.ptp(
        AIC[1], axis=0)), np.max(np.ptp(AIC[2], axis=0)), np.max(np.ptp(AIC[3], axis=0))) / 25.0) * 25

    # Finding proper ylim range for all 4 censored graphs and rounding up
    upper_ylim_censored = int(1 + max(np.max(np.ptp(AIC[4], axis=0)), np.max(np.ptp(
        AIC[5], axis=0)), np.max(np.ptp(AIC[6], axis=0)), np.max(np.ptp(AIC[7], axis=0))) / 25.0) * 25

    upper_ylim = [upper_ylim_uncensored, upper_ylim_censored]

    # Plotting AICs
    for idx, a in enumerate(AIC):
        figure_maker(ax[idx], a, (idx % 4) + 1,
                     upper_ylim[int(idx / 4)], idx > 3)
    subplotLabel(ax)

    return f
