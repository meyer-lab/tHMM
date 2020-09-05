""" This file plots the AIC for the experimental data. """

import numpy as np
from matplotlib.ticker import MaxNLocator
from ..Analyze import run_Analyze_over
import matplotlib.gridspec as gridspec
from ..data.Lineage_collections import Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM

from .figureCommon import getSetup, subplotLabel
from ..tHMM import tHMM
from ..plotTree import plotLineage

desired_num_states = np.arange(1, 5)


def makeFigure():
    """
    Makes figure 9.
    """
    ax, f = getSetup((7, 7), (2, 2))

    data = [Lapatinib_Control + Gemcitabine_Control, Lapt25uM, Lapt50uM, Lap250uM, Gem5uM, Gem10uM, Gem30uM]

    dataFull = []
    for _ in desired_num_states:
        dataFull = dataFull + data

    # Run fitting
    output = run_Analyze_over(dataFull, np.repeat(desired_num_states, len(data)), const=[10, 6])
    AICs = np.array([oo[0].get_AIC(oo[2])[0] for oo in output])
    AICs = np.reshape(AICs, (desired_num_states.size, len(data)))
    AICs -= np.min(AICs, axis=0)
    LAPlins = np.array([oo[0].X[0] for oo in output])[0:3]
    GEMlins = np.array([oo[0].X[0] for oo in output])[-4:-1]

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
    heights = [3, 1, 1, 1]
    widths = [3.5, 3.5]
    spec5 = f.add_gridspec(ncols=2, nrows=4, width_ratios=widths,
                          height_ratios=heights)

    for i in range(2):
        ax[i] = f.add_subplot(spec5[0, i])
        ax[i].plot(desired_num_states, AIC_holder[i])
        ax[i].set_xlabel("Number of States Predicted")
        ax[i].set_ylabel("Normalized AIC")
        ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))

    ax[0].set_title("Lapatinib")
    ax[1].set_title("Gemcitabine")

    i += 1
    ax[i] = f.add_subplot(spec5[1, 0])
    plotLineage(LAPlins[0], ax[i], censore=True)

    # i += 1
    # ax[i] = f.add_subplot(spec5[2, 0])
    # plotLineage()

