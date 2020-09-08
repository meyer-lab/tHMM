""" This file plots the AIC for the experimental data. """

from copy import deepcopy
import numpy as np
from matplotlib.ticker import MaxNLocator
from ..Analyze import run_Analyze_over
import matplotlib.gridspec as gridspec
from ..data.Lineage_collections import Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM

from .figureCommon import getSetup, subplotLabel
from ..tHMM import tHMM

desired_num_states = np.arange(1, 5)


def makeFigure():
    """
    Makes figure 9.
    """
    ax, f = getSetup((7, 3), (1, 2))

    data = [Lapatinib_Control + Gemcitabine_Control, Lapt25uM, Lapt50uM, Lap250uM, Gem5uM, Gem10uM, Gem30uM]
    dataFull = []

    # Find the cell cycle shape parameters to be set as constant from the one state model
    tHMM_solver = tHMM(X=data[0], num_states=1)
    tHMM_solver.fit()

    constant_shape = [int(tHMM_solver.estimate.E[0].params[2]), int(tHMM_solver.estimate.E[0].params[4])]

    # Set shape
    for population in data:
        for lin in population:
            for E in lin.E:
                E.G1.const_shape = constant_shape[0]
                E.G2.const_shape = constant_shape[1]

    # Copy out data to full set
    for _ in desired_num_states:
        dataFull = dataFull + deepcopy(data)

    # Run fitting
    output = run_Analyze_over(dataFull, np.repeat(desired_num_states, len(data)))
    AICs = np.array([oo[0].get_AIC(oo[2])[0] for oo in output])
    AICs = np.reshape(AICs, (desired_num_states.size, len(data)))
    AICs -= np.min(AICs, axis=0)

    lapAIC = np.sum(AICs[:, 0:4], axis=1)
    gemAIC = np.sum(AICs[:, np.array([0, 4, 5, 6])], axis=1)

    # Plotting AICs
    figure_maker(ax, [lapAIC, gemAIC])
    subplotLabel(ax)

    return f


def figure_maker(ax, AIC_holder):
    """
    Makes figure 9.
    """

    for i in range(2):
        ax[i].plot(desired_num_states, AIC_holder[i])
        ax[i].set_xlabel("Number of States Predicted")
        ax[i].set_ylabel("Normalized AIC")
        ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))

    ax[0].set_title("Lapatinib")
    ax[1].set_title("Gemcitabine")
