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
# Lapatinib control pi and T for 2, 3, and 4 states:
fpi_list = [[1.0], [0.66150779, 0.33849221], [6.94307126e-09, 6.99518861e-01, 3.00481132e-01], [6.67672155e-01, 3.32327845e-01, 2.86425040e-16, 4.61223911e-22]]

fT_list = [[1.0], [[0.99277139, 0.00722861],
                   [0.07170348, 0.92829652]], [[9.42386563e-01, 5.76134371e-02, 6.90068935e-28],
                                               [4.77431313e-26, 7.85762403e-01, 2.14237597e-01],
                                               [3.76423244e-01, 1.26698093e-02, 6.10906947e-01]], [[8.13843165e-01, 1.86156835e-01, 2.05422147e-13, 1.54246580e-15],
                                                                                                   [4.65858601e-16, 1.45876129e-01, 1.13605784e-25, 8.54123871e-01],
                                                                                                   [2.88389336e-01, 7.08946381e-01, 2.66398469e-03, 2.98246671e-07],
                                                                                                   [1.19831901e-10, 2.22561316e-06, 9.72941919e-01, 2.70558550e-02]]]


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

    legend = ["cntrl", "25uM", "50uM", "250uM",
              "cntrl", "5uM", "10uM", "30uM"]

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
    output = run_Analyze_AIC(lineages, desired_num_states, const=[10, 6], list_of_fpi=fpi_list, list_if_fT=fT_list)
    for idx in range(len(desired_num_states)):
        AICs[idx], _ = output[idx][0].get_AIC(output[idx][2], 4)

    # Normalizing AIC
    AICs = AICs - np.min(AICs)
    return AICs


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
