"""
Handful of lineages in figure 92.
"""
from .common import getSetup
from ..plotTree import plotLineage_MCF10A
from ..Analyze import Analyze_list
from ..Lineage_collections import pbs, osm

# OSM
OSM = [pbs, osm]
osm_tHMMobj_list, osm_states_list, _ = Analyze_list(OSM, 4, fpi=True)

# assign the predicted states to each cell
for idx, osm_tHMMobj in enumerate(osm_tHMMobj_list):
    for lin_indx, lin in enumerate(osm_tHMMobj.X):
        for cell_indx, cell in enumerate(lin.output_lineage):
            cell.state = osm_states_list[idx][lin_indx][cell_indx]


def makeFigure():
    """
    Makes figure 92 lineage.
    """
    ax, f = getSetup((6, 2), (6, 2))
    for i in range(6):
        ax[i].axis('off')
        plotLineage_MCF10A(osm_tHMMobj_list[0].X[i+4], ax[i])

    for i in range(6, 12):
        ax[i].axis('off')
        plotLineage_MCF10A(osm_tHMMobj_list[1].X[i+6], ax[i])

    return f
