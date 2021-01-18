""" This file plots the AIC for the experimental data. """

import numpy as np
import pickle

from matplotlib.ticker import MaxNLocator
from ..Analyze import run_Analyze_over, Analyze_list
from ..data.Lineage_collections import Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM
from .figureCommon import getSetup, subplotLabel

desired_num_states = np.arange(1, 5)


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
        output = run_Analyze_over(dataFull, desired_num_states, atonce=True)
        AICs = np.array([oo[0][0].get_AIC(oo[2], atonce=True)[0] for oo in output])

        return AICs - np.min(AICs, axis=0)

    lapAIC = find_AIC(lapatinib, desired_num_states)
    gemAIC = find_AIC(gemcitabine, desired_num_states)

    # what is the best number of states
    lpt_st = desired_num_states[np.argmin(lapAIC)]
    gmc_st = desired_num_states[np.argmin(gemAIC)]

    # run analysis for the found number if states
    lapt_tHMMobj_list, lapt_states_list, _ = Analyze_list(lapatinib, lpt_st, fpi=True)
    gemc_tHMMobj_list, gemc_states_list, _ = Analyze_list(gemcitabine, gmc_st, fpi=True)

    # assign the predicted states to each cell
    for idx, lapt_tHMMobj in enumerate(lapt_tHMMobj_list):
        for lin_indx, lin in enumerate(lapt_tHMMobj.X):
            for cell_indx, cell in enumerate(lin.output_lineage):
                cell.state = lapt_states_list[idx][lin_indx][cell_indx]

    for idx, gemc_tHMMobj in enumerate(gemc_tHMMobj_list):
        for lin_indx, lin in enumerate(gemc_tHMMobj.X):
            for cell_indx, cell in enumerate(lin.output_lineage):
                cell.state = gemc_states_list[idx][lin_indx][cell_indx]

    #create a pickle file for lapatinib
    pik1 = open("lapatinibs.pkl", "wb")
    for laps in lapt_tHMMobj_list:
        pickle.dump(laps, pik1)
    pik1.close()

    #create a pickle file for gemcitabine
    pik2 = open("gemcitabines.pkl", "wb")
    for gemc in gemc_tHMMobj_list:
        pickle.dump(gemc, pik2)
    pik2.close()

    # Plotting AICs
    ax[0].plot(desired_num_states, lapAIC)
    ax[1].plot(desired_num_states, gemAIC)

    for i in range(2):
        ax[i].set_xlabel("Number of States Predicted")
        ax[i].set_ylabel("Normalized AIC")
        ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))

    ax[0].set_title("Lapatinib")
    ax[1].set_title("Gemcitabine")

    return f
