""" This file plots the trees with their predicted states for lapatinib. """

from .common import getSetup
from ..plotTree import plotLineage
from ..Analyze import Analyze_list
from ..Lineage_collections import AllGemcitabine

num_states = 5
gemc_tHMMobj_list = Analyze_list(AllGemcitabine, num_states)[0]

gemc_states_list = [tHMMobj.predict() for tHMMobj in gemc_tHMMobj_list]

# assign the predicted states to each cell
for idx, lapt_tHMMobj in enumerate(gemc_tHMMobj_list):
    for lin_indx, lin in enumerate(lapt_tHMMobj.X):
        for cell_indx, cell in enumerate(lin.output_lineage):
            cell.state = gemc_states_list[idx][lin_indx][cell_indx]

only_lapatinib_control_1 = gemc_tHMMobj_list[0].X[0:100]


def makeFigure():
    """
    Makes figure 161.
    """
    ax, f = getSetup((4, 40), (60, 1))

    for i in range(60):
        ax[i].axis('off')
        plotLineage(only_lapatinib_control_1[i], ax[i])

    return f
