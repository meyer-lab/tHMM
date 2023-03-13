""" This file plots the BIC for the experimental data. """
import numpy as np
from matplotlib.ticker import MaxNLocator
from ..Analyze import run_Analyze_over
from ..Lineage_collections import AllLapatinib, AllGemcitabine, GFs
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

    lapBIC, _ = find_BIC(AllLapatinib, desired_num_states, num_cells=5290)
    gemBIC, _ = find_BIC(AllGemcitabine, desired_num_states, num_cells=4537)
    hgfBIC, _ = find_BIC(GFs, desired_num_states, num_cells=1306, mc=True)

    # Plotting BICs
    ax[0].plot(desired_num_states, lapBIC)
    ax[1].plot(desired_num_states, gemBIC)
    ax[2].plot(desired_num_states, hgfBIC)

    for i in range(3):
        ax[i].set_xlabel("Number of States Predicted")
        ax[i].set_ylabel("Normalized BIC")
        ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))

    ax[0].set_title("Lapatinib Treated Populations")
    ax[1].set_title("Gemcitabine Treated Populations")
    ax[2].set_title("Growth Factors Treated Populations")

    return f
