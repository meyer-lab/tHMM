""" Barcoding computational experinece. """

import numpy as np
from .common import getSetup, subplotLabel
from ..Analyze import Analyze_list
from ..Lineage_collections import AllLapatinib, AllGemcitabine

num_lineages = 10


def makeFigure():
    """
    Makes fig barcode.
    """

    # Get list of axis objects
    ax, f = getSetup((15, 12), (4, 5))

    plot_barcode_vs_state(ax[0:10], "lapatinibs")
    plot_barcode_vs_state(ax[10:20], "gemcitabines")
    subplotLabel(ax)

    return f


def plot_barcode_vs_state(ax, drug_name):
    """Plots the histogram of barcode vs states after clustering, using the parameters from lapatinib and gemcitabine fits."""

    if drug_name == "lapatinibs":
        tHMMobj_list, _, _ = Analyze_list(AllLapatinib, 4, write_states=True)
    elif drug_name == "gemcitabines":
        tHMMobj_list, _, _ = Analyze_list(AllGemcitabine, 5, write_states=True)

    num_states = tHMMobj_list[0].num_states

    states_by_lin = []
    for lineage in tHMMobj_list[0].X:
        states_by_lin.append(lineage.states)

    for i in range(num_lineages):
        ax[i].hist(states_by_lin[i], bins=np.linspace(0, 5, 11))
        ax[i].set_title("control, barcode #" + str(i + 1))
        ax[i].set_xticks(np.arange(num_states))
        ax[i].set_xlabel("state #")
        ax[i].set_ylabel("cell #")
