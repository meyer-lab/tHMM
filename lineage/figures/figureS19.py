""" BIC for Taxol data."""
from ..Lineage_collections import taxols as Taxol_lin_list
from ..Analyze import run_Analyze_over, Analyze_list
from .common import getSetup

import pickle
import numpy as np
from matplotlib.ticker import MaxNLocator

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

    BIC, Obj = find_BIC(Taxol_lin_list, desired_num_states, num_cells=1041)

    # create a pickle file
    pik1 = open("taxols.pkl", "wb")
    for lapt_tHMMobj_list in Obj:
        for laps in lapt_tHMMobj_list:
            pickle.dump(laps, pik1)
    pik1.close()

    i = 0
    ax[i].set_xlabel("Number of States Predicted")
    ax[i].set_ylabel("Normalized BIC")
    ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))

    ax[0].set_title("Taxol Treated Populations")

    return f