"""
Handful of lineages in figure 91.
"""
from .common import getSetup
from ..plotTree import plotLineage_MCF10A
from ..Analyze import Analyze_list
from ..Lineage_collections import pbs, hgf

HGF = [pbs, hgf]
hgf_tHMMobj_list, hgf_states_list, _ = Analyze_list(HGF, 5, fpi=True)

# assign the predicted states to each cell
for idx, hgf_tHMMobj in enumerate(hgf_tHMMobj_list):
    for lin_indx, lin in enumerate(hgf_tHMMobj.X):
        for cell_indx, cell in enumerate(lin.output_lineage):
            cell.state = hgf_states_list[idx][lin_indx][cell_indx]


def makeFigure():
    """
    Makes figure 12 lineage.
    """
    ax, f = getSetup((6, 2), (6, 2))

    for i in range(6):
        ax[i].axis('off')
        plotLineage_MCF10A(hgf_tHMMobj_list[0].X[i+4], ax[i])

    for i in range(6, 12):
        ax[i].axis('off')
        plotLineage_MCF10A(hgf_tHMMobj_list[1].X[i+6], ax[i])

    return f
