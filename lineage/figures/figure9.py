""" This file plots the BIC for the experimental data. """
import pickle
import numpy as np
from matplotlib.ticker import MaxNLocator
from ..Analyze import run_Analyze_over, Analyze_list
from ..Lineage_collections import Gemcitabine_Control, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM, Gem5uM, Gem10uM, Gem30uM
from .common import getSetup

desired_num_states = np.arange(1, 8)

def makeFigure():
    """
    Makes figure 9.
    """
    ax, f = getSetup((8, 4), (1, 2))

    lapatinib = [Lapatinib_Control + Gemcitabine_Control, Lapt25uM, Lapt50uM, Lap250uM]
    gemcitabine = [Lapatinib_Control + Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM]

    lapBIC = find_BIC(lapatinib, desired_num_states, num_cells=5290)
    gemBIC = find_BIC(gemcitabine, desired_num_states, num_cells=4537)

    lapt_tHMMobj_list, _ = Analyze_list(lapatinib, list(lapBIC).index(0) + 1, fpi=True)
    lapt_states_list = [tHMMobj.predict() for tHMMobj in lapt_tHMMobj_list]

    # assign the predicted states to each cell
    for idx, lapt_tHMMobj in enumerate(lapt_tHMMobj_list):
        for lin_indx, lin in enumerate(lapt_tHMMobj.X):
            for cell_indx, cell in enumerate(lin.output_lineage):
                cell.state = lapt_states_list[idx][lin_indx][cell_indx]

    # create a pickle file for lapatinib
    pik1 = open("lapatinibs2.pkl", "wb")
    for laps in lapt_tHMMobj_list:
        pickle.dump(laps, pik1)
    pik1.close()

    # Gemcitabine
    gemc_tHMMobj_list, _ = Analyze_list(gemcitabine, list(gemBIC).index(0) + 1, fpi=True)
    gemc_states_list = [tHMMobj.predict() for tHMMobj in gemc_tHMMobj_list]

    for idx, gemc_tHMMobj in enumerate(gemc_tHMMobj_list):
        for lin_indx, lin in enumerate(gemc_tHMMobj.X):
            for cell_indx, cell in enumerate(lin.output_lineage):
                cell.state = gemc_states_list[idx][lin_indx][cell_indx]

    # create a pickle file for gemcitabine
    pik2 = open("gemcitabines2.pkl", "wb")
    for gemc in gemc_tHMMobj_list:
        pickle.dump(gemc, pik2)
    pik2.close()

    # Plotting BICs
    ax[0].plot(desired_num_states, lapBIC)
    ax[0].set_xlabel("Number of States Predicted")
    ax[0].set_ylabel("Normalized BIC")
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].set_title("Lapatinib Treated Populations")

    return f

def find_BIC(data, desired_num_states, num_cells):
    # Copy out data to full set
    dataFull = []
    for _ in desired_num_states:
        dataFull.append(data)

    # Run fitting
    output = run_Analyze_over(dataFull, desired_num_states, atonce=True)
    BICs = np.array([oo[0][0].get_BIC(oo[1], num_cells, atonce=True)[0] for oo in output])

    return BICs - np.min(BICs, axis=0)
