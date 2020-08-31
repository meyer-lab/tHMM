""" This file plots the AIC for the experimental data. """

import numpy as np
from matplotlib.ticker import MaxNLocator
from ..Analyze import run_Analyze_AIC
from ..LineageTree import LineageTree
import matplotlib.gridspec as gridspec
from ..data.Lineage_collections import Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM

# States to evaluate with the model
from ..states.StateDistributionGaPhs import StateDistribution
from .figureCommon import getSetup, subplotLabel

desired_num_states = np.arange(1, 5)

def makeFigure():
    """
    Makes figure 9.
    """
    ax, f = getSetup((7, 3), (1, 2))

    data = [Lapatinib_Control[0:10], Lapt25uM[0:10], Lapt50uM[0:10], Lap250uM[0:10],
            Gemcitabine_Control[0:10], Gem5uM[0:10], Gem10uM[0:10], Gem30uM[0:10]]

    # making lineages and finding AICs (assign number of lineages here)
    AICs = [run_AIC(data[i]) for i in range(len(data))]
    AIC = [np.sum(AICs[0:4], axis=0), np.sum(AICs[4:8], axis=0)]

    # Plotting AICs
    figure_maker(ax, AIC, True)
    subplotLabel(ax)

    return f


def run_AIC(lineages):
    """
    Run AIC for experimental data.
    """

    # Storing AICs into array
    AICs = np.empty(len(desired_num_states))
    output = run_Analyze_AIC(lineages, desired_num_states, const=[10, 6])
    for idx in range(len(desired_num_states)):
        AICs[idx], _ = output[idx][0].get_AIC(output[idx][2], 4)

    # Normalizing AIC
    return AICs - np.min(AICs)


def figure_maker(ax, AIC_holder, censored=False):
    """
    Makes figure 9.
    """

    i = 0
    ax[i].plot(desired_num_states, AIC_holder[0])
    ax[i].set_xlabel("Number of States Predicted")
    ax[i].set_ylabel("Normalized AIC")
    ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[i].set_title("Lapatinib")
    i += 1
    ax[i].plot(desired_num_states, AIC_holder[1])
    ax[i].set_xlabel("Number of States Predicted")
    ax[i].set_ylabel("Normalized AIC")
    ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[i].set_title("Gemcitabine")
