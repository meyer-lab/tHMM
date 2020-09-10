""" This file plots the trees with their predicted states. """

import numpy as np
from matplotlib.ticker import MaxNLocator
import itertools

from ..Analyze import Analyze
from ..data.Lineage_collections import Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM
from .figureCommon import getSetup, subplotLabel
from ..tHMM import tHMM
from ..plotTree import plotLineage


def makeFigure():
    """
    Makes figure 10.
    """
    ax, f = getSetup((7, 12), (8, 2))
    Gem = [Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM]
    Lap = [Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM]

    lapatinib = []
    gemcitabine = []
    # Run fitting
    for indx, data in enumerate(Gem):
        gemc_tHMMobj, gemc_states, _ = Analyze(data, 4) # 4 states predicted by AIC
        lapt_tHMMobj, lapt_states, _ = Analyze(Lap[indx], 3) # 3 states predicted by AIC

        for lin_indx, lin in enumerate(lapt_tHMMobj.X):
            for cell_indx, cell in enumerate(lin.output_lineage):
                cell.state = lapt_states[lin_indx][cell_indx]

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
