""" This file plots the AIC for the experimental data. """

import numpy as np
from matplotlib.ticker import MaxNLocator
from ..Analyze import run_Analyze_over
import matplotlib.gridspec as gridspec
from ..data.Lineage_collections import Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM

from .figureCommon import getSetup, subplotLabel
from ..tHMM import tHMM

desired_num_states = np.arange(1, 3)
Ts = []
PIs = []
# to find the T and pi matrices to be used as the constant and reduce the number of estimations.
for i in desired_num_states:
    tHMM_solver = tHMM(X=Gemcitabine_Control, num_states=i)
    tHMM_solver.fit(const=[22,10])
    # choose the estimated shape parameters for 1-state model to be kept constant
    if i == 1:
        constant_shape = [int(tHMM_solver.estimate.E[0].params[2]), int(tHMM_solver.estimate.E[0].params[4])]
    Ts.append(tHMM_solver.estimate.T)
    PIs.append(tHMM_solver.estimate.pi)


def makeFigure():
    """
    Makes figure 9.
    """
    ax, f = getSetup((7, 7), (4, 2), [3, 1, 1, 1])

    data = [Lapatinib_Control[0:5] + Gemcitabine_Control[0:5], Lapt25uM[0:5], Lapt50uM[0:5], Lap250uM[0:5], Gem5uM[0:5], Gem10uM[0:5], Gem30uM[0:5]]

    dataFull = []
    for _ in desired_num_states:
        dataFull = dataFull + data

    # Run fitting
    output = run_Analyze_over(dataFull, np.repeat(desired_num_states, len(data)), const=[10, 6])
    AICs = np.array([oo[0].get_AIC(oo[2], 4)[0] for oo in output])
    AICs = np.reshape(AICs, (desired_num_states.size, len(data)))
    AICs -= np.min(AICs, axis=0)
    LAPlins = np.array([oo[0].X[0] for oo in output])[np.array([6, 7, 8])]
    GEMlins = np.array([oo[0].X[0] for oo in output])[np.ayyay([-2, -3, -6])]

    lapAIC = np.sum(AICs[:, 0:4], axis=1)
    gemAIC = np.sum(AICs[:, np.array([0, 4, 5, 6])], axis=1)

    # Plotting AICs
    figure_maker(ax, [lapAIC, gemAIC])
    subplotLabel(ax)

    return f


def figure_maker(ax, f, AIC_holder, LAPlins, GEMlins):
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

    # lap
    i += 1
    plotLineage(LAPlins[0], ax[i], censore=True)

    i += 1
    plotLineage(LAPlins[1], ax[i], censore=True)

    i += 1
    plotLineage(LAPlins[2], ax[i], censore=True)

    # gem
    i += 1
    plotLineage(GEMlins[0], ax[i], censore=True)

    i += 1
    plotLineage(GEMlins[1], ax[i], censore=True)

    i += 1
    plotLineage(GEMlins[2], ax[i], censore=True)