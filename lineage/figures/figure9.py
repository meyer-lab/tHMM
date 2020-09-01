""" This file plots the AIC for the experimental data. """

import numpy as np
from matplotlib.ticker import MaxNLocator
from ..Analyze import run_Analyze_over
import matplotlib.gridspec as gridspec
from ..data.Lineage_collections import Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM

from .figureCommon import getSetup, subplotLabel
from ..tHMM import tHMM

desired_num_states = np.arange(1, 5)
Ts = []
PIs = []
# to find the T and pi matrices to be used as the constant and reduce the number of estimations.
for i in desired_num_states:
    tHMM_solver = tHMM(X=Gemcitabine_Control, num_states=i)
    tHMM_solver.fit(const=None)
    Ts.append(tHMM_solver.estimate.T)
    PIs.append(tHMM_solver.estimate.pi)

def makeFigure():
    """
    Makes figure 9.
    """
    ax, f = getSetup((7, 3), (1, 2))

    data = [Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM,
            Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM]

    # making lineages and finding AICs (assign number of lineages here)
    AICs = [run_AIC(data[i]) for i in range(len(data))]
    AIC = [np.sum(AICs[0:4], axis=0), np.sum(AICs[4:8], axis=0)]

    # Plotting AICs
    figure_maker(ax, AIC)
    subplotLabel(ax)

    return f


def run_AIC(lineages):
    """
    Run AIC for experimental data.
    """

    # Storing AICs into array
    output = run_Analyze_over([lineages] * len(desired_num_states), desired_num_states, const=[10, 6], list_of_fpi=PIs, list_if_fT=Ts)
    AICs = [output[idx][0].get_AIC(output[idx][2], 4)[0] for idx in range(len(desired_num_states))]

    # Normalizing AIC
    return np.array(AICs) - np.min(AICs)


def figure_maker(ax, AIC_holder):
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
