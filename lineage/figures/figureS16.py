""" Barcoding computational experinece. """
import numpy as np
import itertools
import pickle
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from copy import deepcopy
from .figureCommon import (
    getSetup,
    subplotLabel,
    num_data_points,
)
from ..LineageTree import LineageTree, max_gen
from ..BaumWelch import calculate_stationary
from ..plotTree import plotLineage
from ..Analyze import run_Analyze_over
from ..states.StateDistributionGaPhs import StateDistribution


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

def cenGenBarcode(num, pi, T, E, i):
    tmp = LineageTree.init_from_parameters(pi, T, E, desired_num_cells=num, barcode=i, censor_condition=3, desired_experiment_time=100)
    while len(tmp.output_lineage) < 3:
        tmp = LineageTree.init_from_parameters(pi, T, E, desired_num_cells=num, barcode=i, censor_condition=3, desired_experiment_time=100)
    return tmp

def plot_barcode_vs_state(ax, drug_name):
    """ Plots the histogram of barcode vs states after clustering, using the parameters from lapatinib and gemcitabine fits. """
    pik = open(str(drug_name) +".pkl", "rb")
    tHMMobj_list = []
    for i in range(4):
        tHMMobj_list.append(pickle.load(pik))
    T = tHMMobj_list[0].estimate.T
    pi = calculate_stationary(T)
    E = tHMMobj_list[0].estimate.E
    num_states = tHMMobj_list[0].num_states
    list_of_fpi = [pi] * num_lineages

    # Adding populations into a holder for analysing
    population = [[cenGenBarcode(23, pi, T, E, i+1)] for i in range(num_lineages)]
    output = run_Analyze_over(population, num_states, parallel=True, list_of_fpi=list_of_fpi)
    print(len(output))

    barcodes_by_lin = []
    states_by_lin = []
    for tHMMobj in output:
        tmp1 = []
        tmp2 = []
        for cell in output[0][0].X[0].output_lineage:
            for cell in lineage.output_lineage:
                tmp1.append(cell.barcode)
                tmp2.append(cell.state)
        barcodes_by_lin.append(tmp1)
        states_by_lin.append(tmp2)
    print(len(states_by_lin))

    assert len(states_by_lin) == len(barcodes_by_lin)

    for i in range(num_lineages):
        ax[i].hist(states_by_lin[i])
        ax[i].set_title(str(drug_name) + ", barcode #" + str(i+1))
        ax[i].set_xticks(np.arange(num_states))
        ax[i].set_xlabel("state #")
        ax[i].set_ylabel("cell #")








    # all_conditions = []
    # for tHMM in tHMM_list:
    #     counts_per_lin = np.zeros((num_states, 5))
    #     for lineage in tHMM.X:
    #         # for each lineage, count how many cells are in each state in each generation
    #         for i, cells_ls in enumerate(lineage.output_list_of_gens[1:]):
    #             assert cells_ls[0] is not None
    #             sts = [cell.state for cell in cells_ls]
    #             counts_per_lin[:, i] += [sts.count(t) for t in range(num_states)]
    #     all_conditions.append(counts_per_lin)

