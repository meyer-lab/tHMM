""" BIC foe MCF10A data. """

import numpy as np

from matplotlib.ticker import MaxNLocator
from ..Analyze import run_Analyze_over
from ..Lineage_collections import pbs, egf, hgf, osm
from .common import getSetup

desired_num_states = np.arange(1, 8)

def find_BIC(data, desired_num_states, num_cells):
    # Copy out data to full set
    dataFull = []
    for _ in desired_num_states:
        dataFull.append(data)

    # Run fitting
    output = run_Analyze_over(dataFull, desired_num_states, atonce=True)
    BICs = np.array([oo[0][0].get_BIC(oo[2], num_cells, atonce=True)[0] for oo in output])

    return BICs - np.min(BICs, axis=0)

def makeFigure():
    """
    Makes figure 90.
    """
    ax, f = getSetup((8, 3), (1, 2))

    # cell numbers: pbs: 31, egf: 255, hgf: 507, osm: 692
    # after removing single lineages [262, 503, 695]
    HGF = [pbs, hgf]
    OSM = [pbs, osm]

    hgfBIC = find_BIC(HGF, desired_num_states, num_cells=538)
    osmBIC = find_BIC(OSM, desired_num_states, num_cells=723)

    # Plotting BICs
    ax[0].plot(desired_num_states, hgfBIC)
    ax[1].plot(desired_num_states, osmBIC)

    for i in range(2):
        ax[i].set_xlabel("Number of States Predicted")
        ax[i].set_ylabel("Normalized BIC")
        ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))

    ax[0].set_title("HGF Treated Populations")
    ax[1].set_title("OSM Treated Populations")

    return f
