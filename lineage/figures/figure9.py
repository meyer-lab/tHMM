""" This file plots the AIC for the experimental data. """

from copy import deepcopy
import numpy as np
from matplotlib.ticker import MaxNLocator
from ..Analyze import run_Analyze_over
import matplotlib.gridspec as gridspec
from ..data.Lineage_collections import Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM

from .figureCommon import getSetup, subplotLabel
from ..tHMM import tHMM

desired_num_states = np.arange(1, 8)


def makeFigure():
    """
    Makes figure 9.
    """
    ax, f = getSetup((7, 3), (1, 2))

    lapatinib = [Lapatinib_Control + Gemcitabine_Control, Lapt25uM, Lapt50uM, Lap250uM]
    gemcitabine = [Lapatinib_Control + Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM]

    def find_AIC(data, desired_num_states):
        # Copy out data to full set
        dataFull = []
        for _ in desired_num_states:
            dataFull.append(data)

        # Run fitting
        output = run_Analyze_over(dataFull, desired_num_states)
        AICs = np.array([oo[0][0].get_AIC(oo[2], atonce=True)[0] for oo in output])
        AICs -= np.min(AICs, axis=0)

        return AICs

    lapAIC = find_AIC(lapatinib, desired_num_states)
    gemAIC = find_AIC(gemcitabine, desired_num_states)

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
