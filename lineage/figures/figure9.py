""" This file plots the BIC for the experimental data. """

import numpy as np
import pickle
from matplotlib.ticker import MaxNLocator
from ..Analyze import run_Analyze_over, Analyze_list
from ..Lineage_collections import Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM
from .common import getSetup

desired_num_states = np.arange(1, 8)


def makeFigure():
    """
    Makes figure 9.
    """
    ax, f = getSetup((9, 4), (1, 2))

    lapatinib = [Lapatinib_Control + Gemcitabine_Control, Lapt25uM, Lapt50uM, Lap250uM]
    gemcitabine = [Lapatinib_Control + Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM]

    def find_BIC(data, desired_num_states, num_cells):
        # Copy out data to full set
        dataFull = []
        for _ in desired_num_states:
            dataFull.append(data)

        # Run fitting
        output = run_Analyze_over(dataFull, desired_num_states, atonce=True)
        BICs = np.array([oo[0][0].get_BIC(oo[1], num_cells, atonce=True)[0] for oo in output])

        return BICs - np.min(BICs, axis=0)

    lapBIC = find_BIC(lapatinib, desired_num_states, num_cells=5290)
    gemBIC = find_BIC(gemcitabine, desired_num_states, num_cells=4537)

    # # Lapatinib
    # lapt_tHMMobj_list, _ = Analyze_list(lapatinib, 6, fpi=True)
    # lapt_states_list = [tHMMobj.predict() for tHMMobj in lapt_tHMMobj_list]

    # # assign the predicted states to each cell
    # for idx, lapt_tHMMobj in enumerate(lapt_tHMMobj_list):
    #     for lin_indx, lin in enumerate(lapt_tHMMobj.X):
    #         for cell_indx, cell in enumerate(lin.output_lineage):
    #             cell.state = lapt_states_list[idx][lin_indx][cell_indx]

    # # create a pickle file for lapatinib
    # pik1 = open("lapatinibs.pkl", "wb")
    # for laps in lapt_tHMMobj_list:
    #     pickle.dump(laps, pik1)
    # pik1.close()

    # # Gemcitabine
    # gemc_tHMMobj_list, _ = Analyze_list(gemcitabine, 5, fpi=True)
    # gemc_states_list = [tHMMobj.predict() for tHMMobj in gemc_tHMMobj_list]

    # for idx, gemc_tHMMobj in enumerate(gemc_tHMMobj_list):
    #     for lin_indx, lin in enumerate(gemc_tHMMobj.X):
    #         for cell_indx, cell in enumerate(lin.output_lineage):
    #             cell.state = gemc_states_list[idx][lin_indx][cell_indx]

    # # create a pickle file for gemcitabine
    # pik2 = open("gemcitabines.pkl", "wb")
    # for gemc in gemc_tHMMobj_list:
    #     pickle.dump(gemc, pik2)
    # pik2.close()

    # Plotting BICs
    ax[0].plot(desired_num_states, lapBIC)
    ax[1].plot(desired_num_states, gemBIC)

    for i in range(2):
        ax[i].set_xlabel("Number of States Predicted")
        ax[i].set_ylabel("Normalized BIC")
        ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))

    ax[0].set_title("Lapatinib Treated Populations")
    ax[1].set_title("Gemcitabine Treated Populations")

    return f
