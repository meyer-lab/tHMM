""" This file plots the trees with their predicted states for growth factors. """

import numpy as np
from .common import getSetup, sort_lins
from ..plotTree import plotLineage_MCF10A

from ..Lineage_collections import GFs
from ..Analyze import Analyze_list

hgf_tHMMobj_list, _,_ = Analyze_list(GFs, 3)
hgf_states_list = [tHMMobj.predict() for tHMMobj in hgf_tHMMobj_list]

# assign the predicted states to each cell
for idx, hgf_tHMMobj in enumerate(hgf_tHMMobj_list):
    for lin_indx, lin in enumerate(hgf_tHMMobj.X):
        for cell_indx, cell in enumerate(lin.output_lineage):
            cell.state = hgf_states_list[idx][lin_indx][cell_indx]

for thmm_obj in hgf_tHMMobj_list:
    thmm_obj.X = sort_lins(thmm_obj)

conditions = ["PBS", "EGF", "HGF", "OSM"]


def makeFigure():
    """
    Makes figure 101.
    """
    num_lins = [len(hgf_tHMMobj_list[i].X) for i in range(4)]
    ax, f = getSetup((10, 20), (np.max(num_lins), 4))

    for i in range(4 * np.max(num_lins)):
        ax[i].axis('off')

    for j, thmmobj in enumerate(hgf_tHMMobj_list):
        for i, X in enumerate(thmmobj.X):
            plotLineage_MCF10A(X, ax[4 * i + j])

    for i in range(4):
        ax[i].set_title(conditions[i])

    return f
