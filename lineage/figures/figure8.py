""" This file plots the AIC for the real data. """

import numpy as np
from matplotlib.ticker import MaxNLocator
from ..Analyze import run_Analyze_AIC
from ..LineageTree import LineageTree
import matplotlib.gridspec as gridspec
from ..data.Lineage_collections import gemControl, gem5uM, lap25uM, taxs

# States to evaluate with the model
from ..states.StateDistributionGaPhs import StateDistribution
from .figureCommon import getSetup, subplotLabel
from .figureS11 import run_AIC, figure_maker


desired_num_states = np.arange(1, 5)


def makeFigure():
    """
    Makes figure 8.
    """
    ax, f = getSetup((13.333, 3.333), (1, 4))

    # making lineages and finding AICs (assign number of lineages here)
    AIC = [run_AIC(gemControl[1:9]), run_AIC(gem5uM[1:9]), run_AIC(lap25uM[1:9]), run_AIC(taxs[1:9])]

    # Finding proper ylim range for all 4 censored graphs and rounding up
    upper_ylim_censored = int(1 + max(np.max(np.ptp(AIC[0], axis=0)), np.max(np.ptp(
        AIC[1], axis=0)), np.max(np.ptp(AIC[2], axis=0)), np.max(np.ptp(AIC[3], axis=0))) / 25.0) * 25

    upper_ylim = [upper_ylim_censored]
    titles = ["Cntrl", "Gem 5uM", "Lapt 25uM", "Tax 2uM"]

    # Plotting AICs
    for idx, a in enumerate(AIC):
        figure_maker(ax[idx], a, titles[idx],
                     upper_ylim[0], True)
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

    print(AICs)
    return AICs

def figure_maker(ax, AIC_holder, title, upper_ylim, censored=False):
    """
    Makes figure 10.
    """
    # Normalizing AIC
    AIC_holder = AIC_holder - np.min(AIC_holder, axis=0)[np.newaxis, :]

    # Creating Histogram and setting ylim
    ax2 = ax.twinx()
    ax2.set_ylabel("Lineages Predicted")
    ax2.hist(np.argmin(AIC_holder, axis=0) + 1, rwidth=1,
             alpha=.2, bins=desired_num_states, align='left')
    ax2.set_yticks(np.linspace(0, len(AIC_holder[0]), 1 + len(AIC_holder[0])))

    # Creating AIC plot and matching gridlines
    ax.set_xlabel("Number of States Predicted")
    ax.plot(desired_num_states, AIC_holder, "k", alpha=0.5)
    ax.set_ylabel("Normalized AIC")
    ax.set_yticks(np.linspace(0, upper_ylim, len(ax2.get_yticks())))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Adding title
    title = f"AIC for {title} "
    ax.set_title(title)
