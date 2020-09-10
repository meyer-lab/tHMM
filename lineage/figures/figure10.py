""" This file plots the trees with their predicted states. """

import numpy as np
from matplotlib.ticker import MaxNLocator
import itertools

from ..Analyze import run_Analyze_over
from ..data.Lineage_collections import Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM
from .figureCommon import getSetup, subplotLabel
from ..tHMM import tHMM
from ..plotTree import plotLineage


def makeFigure():
    """
    Makes figure 10.
    """
    ax, f = getSetup((7, 12), (8, 2))
    data = [Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM]

    lapatinib = []
    gemcitabine = []
    # Run fitting
    output = run_Analyze_over(data, np.repeat([3, 4], 4))
    gemc_tHMMobj_list = [output[i][0] for i in range(5)]
    gemc_states_list = [output[i][1] for i in range(5)]
    lapt_tHMMobj_list = [output[i][0] for i in range(4, 8)]
    lapt_states_list = [output[i][1] for i in range(4, 8)]

    for idx, lapt_tHMMobj in enumerate(lapt_tHMMobj_list):
        for lin_indx, lin in enumerate(lapt_tHMMobj.X):
            for cell_indx, cell in enumerate(lin.output_lineage):
                cell.state = lapt_states_list[idx][lin_indx][cell_indx]
    for idx, gemc_tHMMobj in enumerate(gemc_tHMMobj_list):
        for lin_indx, lin in enumerate(gemc_tHMMobj.X):
            for cell_indx, cell in enumerate(lin.output_lineage):
                cell.state = gemc_states[lin_indx][cell_indx]
        lapatinib.append([lapt_tHMMobj.X[4], lapt_tHMMobj.X[7]])
        gemcitabine.append([gemc_tHMMobj.X[0], gemc_tHMMobj.X[3]])

    # Plotting the lineages
    figure_maker(ax, list(itertools.chain(*lapatinib)), list(itertools.chain(*gemcitabine)))

    return f


def figure_maker(ax, lapatinib, gemcitabine):
    """
    Makes figure 10.
    """

    ax[0].set_title("Lapatinib")
    ax[1].set_title("Gemcitabine")

    i = 0
    for j in np.arange(0, 15, 2):
        ax[j].axis('off')
        plotLineage(lapatinib[i], ax[j], censore=True)
        i += 1
    i = 0
    for j in np.arange(1, 16, 2):
        ax[j].axis('off')
        plotLineage(gemcitabine[i], ax[j], censore=True)
        i += 1
