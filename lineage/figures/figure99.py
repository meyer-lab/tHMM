""" This file plots the BIC for the experimental data. """
import pickle
import numpy as np
from matplotlib.ticker import MaxNLocator
from ..Analyze import run_Analyze_over, Analyze_list
from ..Lineage_collections import Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM, Lapatinib_Control
from .common import getSetup
from .figure9 import find_BIC

desired_num_states = np.arange(1, 8)

def makeFigure():
    """
    Makes figure 9.
    """
    ax, f = getSetup((4, 4), (1, 1))

    gemcitabine = [Lapatinib_Control + Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM]

    gemc_tHMMobj_list, _ = Analyze_list(gemcitabine, 5, fpi=True)
    gemc_states_list = [tHMMobj.predict() for tHMMobj in gemc_tHMMobj_list]

    for idx, gemc_tHMMobj in enumerate(gemc_tHMMobj_list):
        for lin_indx, lin in enumerate(gemc_tHMMobj.X):
            for cell_indx, cell in enumerate(lin.output_lineage):
                cell.state = gemc_states_list[idx][lin_indx][cell_indx]

    # create a pickle file for gemcitabine
    pik2 = open("gemcitabines.pkl", "wb")
    for gemc in gemc_tHMMobj_list:
        pickle.dump(gemc, pik2)
    pik2.close()

    # gemBIC, output2 = find_BIC(gemcitabine, desired_num_states, num_cells=4537)

    # Gemcitabine
    gemc_tHMMobj_list = output2[list(gemBIC).index(0)][0]
    gemc_states_list = [tHMMobj.predict() for tHMMobj in gemc_tHMMobj_list]

    for idx, gemc_tHMMobj in enumerate(gemc_tHMMobj_list):
        for lin_indx, lin in enumerate(gemc_tHMMobj.X):
            for cell_indx, cell in enumerate(lin.output_lineage):
                cell.state = gemc_states_list[idx][lin_indx][cell_indx]

    # create a pickle file for gemcitabine
    pik2 = open("gemcitabines.pkl", "wb")
    for gemc in gemc_tHMMobj_list:
        pickle.dump(gemc, pik2)
    pik2.close()

    # Plotting BICs
    ax[0].plot(desired_num_states, gemBIC)
    ax[0].set_xlabel("Number of States Predicted")
    ax[0].set_ylabel("Normalized BIC")
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].set_title("Gemcitabine Treated Populations")

    return f
