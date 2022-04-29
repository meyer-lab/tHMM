""" BIC for MCF10A data. """

import numpy as np
import pickle

from matplotlib.ticker import MaxNLocator
from ..Analyze import run_Analyze_over, Analyze_list
from ..Lineage_collections import pbs, egf, hgf, osm
from .common import getSetup

desired_num_states = np.arange(1, 8)
GFs = [pbs, egf, hgf, osm]


def find_BIC(data, desired_num_states, num_cells):
    # Copy out data to full set
    dataFull = []
    for _ in desired_num_states:
        dataFull.append(data)

    # Run fitting
    output = run_Analyze_over(dataFull, desired_num_states, atonce=True)
    BICs = np.array([oo[0][0].get_BIC(oo[1], num_cells, atonce=True, mcf10a=True)[0] for oo in output])

    return BICs - np.min(BICs, axis=0)


hgfBIC = find_BIC(GFs, desired_num_states, num_cells=1306)

# HGF
hgf_tHMMobj_list, hgf_states_list, _ = Analyze_list(GFs, list(hgfBIC).index(0) + 1, fpi=True)

hgf_states_list = [tHMMobj.predict() for tHMMobj in hgf_tHMMobj_list]

# assign the predicted states to each cell
for idx, hgf_tHMMobj in enumerate(hgf_tHMMobj_list):
    for lin_indx, lin in enumerate(hgf_tHMMobj.X):
        for cell_indx, cell in enumerate(lin.output_lineage):
            cell.state = hgf_states_list[idx][lin_indx][cell_indx]

# create a pickle file for osm
pik1 = open("gf.pkl", "wb")
for hgfd in hgf_tHMMobj_list:
    pickle.dump(hgfd, pik1)
pik1.close()


def makeFigure():
    """
    Makes figure 90.
    """
    ax, f = getSetup((4, 4), (1, 1))

    # cell numbers: pbs: 31, egf: 76, hgf: 507, osm: 692
    # after removing single lineages [262, 503, 695]

    # Plotting BICs
    ax[0].plot(desired_num_states, hgfBIC)
    ax[0].set_xlabel("Number of States Predicted")
    ax[0].set_ylabel("Normalized BIC")
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].set_title("GF Treated Populations")

    return f
