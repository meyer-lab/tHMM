""" This file plots the BIC for the experimental data. """
import pickle
import numpy as np
from matplotlib.ticker import MaxNLocator
from ..Analyze import run_Analyze_over, Analyze_list
from ..Lineage_collections import Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM, pbs, egf, hgf, osm
from .common import getSetup

desired_num_states = np.arange(1, 8)

def find_BIC(data, desired_num_states, num_cells, mc=False):
    # Copy out data to full set
    dataFull = []
    for _ in desired_num_states:
        dataFull.append(data)

    # Run fitting
    output = run_Analyze_over(dataFull, desired_num_states, atonce=True)
    BICs = np.array([oo[0][0].get_BIC(oo[1], num_cells, atonce=True, mcf10a=mc)[0] for oo in output])
    thobj = [oo[0] for oo in output]

    return BICs - np.min(BICs, axis=0), thobj

def makeFigure():
    """
    Makes figure 9.
    """
    ax, f = getSetup((14, 4), (1, 3))

    lapatinib = [Lapatinib_Control + Gemcitabine_Control, Lapt25uM, Lapt50uM, Lap250uM]
    gemcitabine = [Lapatinib_Control + Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM]
    GFs = [pbs, egf, hgf, osm]

    # lapBIC, lapObj = find_BIC(lapatinib, desired_num_states, num_cells=5290)
    # gemBIC, gemObj = find_BIC(gemcitabine, desired_num_states, num_cells=4537)
    hgfBIC, hgfObj = find_BIC(GFs, desired_num_states, num_cells=1306, mc=True)

    # Lapatinib
    # # create a pickle file for lapatinib
    # pik1 = open("lapatinibs.pkl", "wb")
    # for lapt_tHMMobj_list in lapObj:
    #     for laps in lapt_tHMMobj_list:
    #         pickle.dump(laps, pik1)
    # pik1.close()

    # # Gemcitabine
    # pik2 = open("gemcitabines.pkl", "wb")
    # for gemc_tHMMobj_list in gemObj:
    #     for gemc in gemc_tHMMobj_list:
    #         pickle.dump(gemc, pik2)
    # pik2.close()

    # Growth factors
    pik3 = open("gf.pkl", "wb")
    for hgf_tHMMobj_list in hgfObj:
        for hgfd in hgf_tHMMobj_list:
            pickle.dump(hgfd, pik3)
    pik3.close()

    # Plotting BICs
    # ax[0].plot(desired_num_states, lapBIC)
    # ax[1].plot(desired_num_states, gemBIC)
    ax[2].plot(desired_num_states, hgfBIC)

    for i in range(3):
        ax[i].set_xlabel("Number of States Predicted")
        ax[i].set_ylabel("Normalized BIC")
        ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))

    ax[0].set_title("Lapatinib Treated Populations")
    ax[1].set_title("Gemcitabine Treated Populations")
    ax[2].set_title("Growth Factors Treated Populations")

    return f
