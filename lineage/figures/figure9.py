""" This file plots the AIC for the experimental data. """

import numpy as np
from matplotlib.ticker import MaxNLocator
from ..Analyze import run_Analyze_over
import matplotlib.gridspec as gridspec
from ..data.Lineage_collections import Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM

from .figureCommon import getSetup, subplotLabel
from ..tHMM import tHMM

desired_num_states = np.arange(1, 7)
Ts = []
PIs = []
# to find the T and pi matrices to be used as the constant and reduce the number of estimations.
for i in desired_num_states:
    tHMM_solver = tHMM(X=Gemcitabine_Control, constant_params=None, num_states=i)
    tHMM_solver.fit()
    # choose the estimated shape parameters for 1-state model to be kept constant
    if i == 1:
        constant_shape = [tHMM_solver.estimate.E[0].params[2], tHMM_solver.estimate.E[0].params[4]]
    Ts.append(tHMM_solver.estimate.T)
    PIs.append(tHMM_solver.estimate.pi)


def makeFigure():
    """
    Makes figure 9.
    """
    ax, f = getSetup((7, 3), (1, 2))

    data = [Lapatinib_Control + Gemcitabine_Control, Lapt25uM, Lapt50uM, Lap250uM, Gem5uM, Gem10uM, Gem30uM]

    dataFull = []
    for _ in desired_num_states:
        dataFull = dataFull + data

    # Run fitting
    output = run_Analyze_over(dataFull, np.repeat(desired_num_states, len(data)), constant_params=constant_shape)
    AICs = np.array([oo[0].get_AIC(oo[2], 4)[0] for oo in output])
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
