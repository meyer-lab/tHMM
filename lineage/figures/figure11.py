""" This file depicts the distribution of phase lengths versus the states. """
import numpy as np
from matplotlib.ticker import MaxNLocator
import itertools

from ..Analyze import run_Analyze_over
from ..data.Lineage_collections import Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM
from .figureCommon import getSetup, subplotLabel


def makeFigure():
    """
    Makes figure 10.
    """
    ax, f = getSetup((7, 6), (2, 2))
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
            level1 = []
            for cell_indx, cell in enumerate(lin.output_lineage):
                cell.state = lapt_states_list[idx][lin_indx][cell_indx]
                lapatinib.append([cell.state, cell.obs[2], cell.obs[3]])

    for idx, gemc_tHMMobj in enumerate(gemc_tHMMobj_list):
        for lin_indx, lin in enumerate(gemc_tHMMobj.X):
            for cell_indx, cell in enumerate(lin.output_lineage):
                cell.state = gemc_states_list[idx][lin_indx][cell_indx]
                gemcitabine.append([cell.state, cell.obs[2], cell.obs[3]])

    # plot
    for i in range(2):
        ax[i].scatter([a[0] for a in lapatinib], [a[i+1] for a in lapatinib], alpha=0.3, marker="+", c="#00ffff")
        ax[i].set_ylabel("G1 phase lengths")
        ax[i].set_xlabel("state")
        ax[i].set_title("Lapatinib treatment")
        ax[i].set_ylim(bottom=-20, top=180)
        ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[1].set_ylabel("G2 phase lengths")

    for i in range(2, 4):
        ax[i].scatter([a[0] for a in gemcitabine], [a[i-1] for a in gemcitabine], alpha=0.3, marker="+", c="#feba4f")
        ax[i].set_ylabel("G1 phase lengths")
        ax[i].set_xlabel("state")
        ax[i].set_title("Gemcitabine treatment")
        ax[i].set_ylim(bottom=-20, top=180)
        ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[3].set_ylabel("G2 phase lengths")

    return f
