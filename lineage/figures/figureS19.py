""" BIC for Taxol data."""
from ..Lineage_collections import taxols as Taxol_lin_list
from ..Analyze import run_Analyze_over, Analyze_list
from .common import getSetup
from .figure9 import find_BIC

import pickle
import numpy as np
from matplotlib.ticker import MaxNLocator

desired_num_states = np.arange(1, 8)

def makeFigure():
    """
    Makes figure 9.
    """
    ax, f = getSetup((14, 4), (1, 3))

    BIC, Obj = find_BIC(Taxol_lin_list, desired_num_states, num_cells=1041)

    # create a pickle file
    pik1 = open("taxols.pkl", "wb")
    for taxol_tHMMobj_list in Obj:
        for laps in taxol_tHMMobj_list:
            pickle.dump(laps, pik1)
    pik1.close()

    i = 0
    ax[i].set_xlabel("Number of States Predicted")
    ax[i].set_ylabel("Normalized BIC")
    ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))

    ax[0].set_title("Taxol Treated Populations")

    return f