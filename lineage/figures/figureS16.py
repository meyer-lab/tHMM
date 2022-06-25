""" Barcoding computational experinece. """
import numpy as np
import pickle
from .common import (
    getSetup,
    subplotLabel
)


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
    """ Plots the histogram of barcode vs states after clustering, using the parameters from lapatinib and gemcitabine fits. """
    pik1 = open(str(drug_name) + ".pkl", "rb")
    alls = []
    for i in range(7):
        tHMMobj_list = []
        for i in range(4):
            tHMMobj_list.append(pickle.load(pik1))
        alls.append(tHMMobj_list)

    # selected for gemcitabine is 5 states which is index 4.
    tHMMobj_list = alls[4]

    states_list = [tHMMobj.predict() for tHMMobj in tHMMobj_list]

    for idx, tHMMobj in enumerate(tHMMobj_list):
        for lin_indx, lin in enumerate(tHMMobj.X):
            for cell_indx, cell in enumerate(lin.output_lineage):
                cell.state = states_list[idx][lin_indx][cell_indx]

    num_states = tHMMobj_list[0].num_states

    states_by_lin = []
    for lineage in tHMMobj_list[0].X:
        tmp2 = []
        for cell in lineage.output_lineage:
            tmp2.append(cell.state)
        states_by_lin.append(tmp2)

    for i in range(num_lineages):
        ax[i].hist(states_by_lin[i], bins=np.linspace(0, 5, 11))
        ax[i].set_title("control, barcode #" + str(i + 1))
        ax[i].set_xticks(np.arange(num_states))
        ax[i].set_xlabel("state #")
        ax[i].set_ylabel("cell #")
