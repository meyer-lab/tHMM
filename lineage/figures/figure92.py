""" Plotting the results for HGF. """
""" This file depicts the distribution of phase lengths versus the states for each concentration of lapatinib. """
from string import ascii_lowercase
import numpy as np

from .common import getSetup
from .figure91 import plot2
from ..Analyze import Analyze_list
from ..Lineage_collections import pbs, osm
from ..plotTree import plot_networkx

OSM = [pbs, osm]
concs = concsValues = ["PBS", "OSM"]

# OSM
osm_tHMMobj_list, osm_states_list, _ = Analyze_list(OSM, 4, fpi=True)

# assign the predicted states to each cell
for idx, osm_tHMMobj in enumerate(osm_tHMMobj_list):
    for lin_indx, lin in enumerate(osm_tHMMobj.X):
        for cell_indx, cell in enumerate(lin.output_lineage):
            cell.state = osm_states_list[idx][lin_indx][cell_indx]

T_osm = osm_tHMMobj_list[0].estimate.T
num_states = osm_tHMMobj_list[0].num_states

def makeFigure():
    """ Makes figure 11. """

    ax, f = getSetup((15, 6), (2, 4))
    plot2(ax, num_states, osm_tHMMobj_list, "OSM", concs, concsValues)
    for i in range(2, 4):
        ax[i].set_title(concs[i - 2], fontsize=16)
        ax[i].text(-0.2, 1.25, ascii_lowercase[i - 2], transform=ax[i].transAxes, fontsize=16, fontweight="bold", va="top")
        ax[i].axis('off')
    # plot_networkx(4, T_osm, "OSM")

    return f
