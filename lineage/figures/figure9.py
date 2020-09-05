""" This file plots the AIC for the experimental data. """

import numpy as np
from matplotlib.ticker import MaxNLocator
from ..Analyze import run_Analyze_over
import matplotlib.gridspec as gridspec
from ..data.Lineage_collections import Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM

from .figureCommon import getSetup, subplotLabel
from ..tHMM import tHMM

desired_num_states = np.arange(1, 8)
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
    ax, f = getSetup((7, 14), (4, 2))

    data = [Lapatinib_Control + Gemcitabine_Control, Lapt25uM, Lapt50uM, Lap250uM, Gem5uM, Gem10uM, Gem30uM]

    dataFull = []
    for _ in desired_num_states:
        dataFull = dataFull + data

    output = run_Analyze_over(dataFull, np.repeat(desired_num_states, len(data)), list_of_fpi=np.repeat(PIs, len(data)).tolist(), list_of_fT=np.repeat(Ts, len(data)).tolist())
    AICs = np.array([oo[0].get_AIC(oo[2], 4)[0] for oo in output])
    AICs = np.reshape(AICs, (desired_num_states.size, len(data)))
    AICs -= np.min(AICs, axis=0)
    LAPlins = np.array([oo[0].X[0] for oo in output])[np.array([6, 7, 17])]
    GEMlins = np.array([oo[0].X[0] for oo in output])[np.ayyay([-2, -3, -6])]

    lapAIC = np.sum(AICs[:, 0:4], axis=1)
    gemAIC = np.sum(AICs[:, np.array([0, 4, 5, 6])], axis=1)

    # Plotting AICs
    figure_maker(ax, f, [lapAIC, gemAIC], LAPlins, GEMlins)
    subplotLabel(ax)

    return f


def figure_maker(ax, f, AIC_holder, LAPlins, GEMlins):
    """
    Makes figure 9.
    """
    # heights = [3, 1, 1, 1]
    # widths = [3.5, 3.5]
    # spec5 = f.add_gridspec(ncols=2, nrows=4, width_ratios=widths,
    #                       height_ratios=heights)

    for i in range(2):
        # ax[i] = f.add_subplot(spec5[0, i])
        ax[i].plot(desired_num_states, AIC_holder[i])
        ax[i].set_xlabel("Number of States Predicted")
        ax[i].set_ylabel("Normalized AIC")
        ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))

    ax[0].set_title("Lapatinib")
    ax[1].set_title("Gemcitabine")

    # lap
    i += 1
    # ax[i] = f.add_subplot(spec5[1, 0])
    plotLineage(LAPlins[0], ax[i], censore=True)

    i += 1
    # ax[i] = f.add_subplot(spec5[2, 0])
    plotLineage(LAPlins[1], ax[i], censore=True)

    i += 1
    # ax[i] = f.add_subplot(spec5[3, 0])
    plotLineage(LAPlins[2], ax[i], censore=True)

    # gem
    i += 1
    # ax[i] = f.add_subplot(spec5[1, 1])
    plotLineage(GEMlins[0], ax[i], censore=True)

    i += 1
    # ax[i] = f.add_subplot(spec5[2, 1])
    plotLineage(GEMlins[1], ax[i], censore=True)

    i += 1
    # ax[i] = f.add_subplot(spec5[3, 1])
    plotLineage(GEMlins[2], ax[i], censore=True)