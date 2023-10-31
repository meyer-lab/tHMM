""" This file includes functions to separate cell states of different replicates, and plot the barchart for each condition."""

import numpy as np
from .common import getSetup
from lineage.Lineage_collections import (
    AllLapatinib,
    AllGemcitabine,
    lpt_cn_reps,
    lpt_25_reps,
    lpt_50_reps,
    lpt_250_reps,
    gem_cn_reps,
    gem_5_reps,
    gem_10_reps,
    gem_30_reps,
)
from ..Analyze import Analyze_list

lapt_tHMMobj_list = Analyze_list(AllLapatinib, 4)[0]
gemc_tHMMobj_list = Analyze_list(AllGemcitabine, 5)[0]


def convertToIndex(lpt_cn_reps):
    """Takes the list containing the lineage numbers of each replicate,
    and returning the list of their index correspondance."""
    for i, val in enumerate(lpt_cn_reps):
        if i == 0:
            lpt_cn_reps[i] = val
        else:
            lpt_cn_reps[i] = lpt_cn_reps[i] + lpt_cn_reps[i - 1]
    return [0] + lpt_cn_reps


def separate_reps(rep_indx_list, thmm):
    """For a given condition, ie., control, 25nM, etc.,
    makes a list of lists containing lineage_trees of separate replicates."""
    reps = []
    for k, val in enumerate(rep_indx_list):
        reps.append([thmm.X[val : rep_indx_list[k + 1]]])
        if k >= 2:
            break
    assert len(reps) == 3
    return reps


def state_abundance_perRep(reps):
    """Finds the number of cells in each state for all replicates of a condition."""
    s0 = []
    s1 = []
    s2 = []
    s3 = []
    s4 = []
    s5 = []
    for rep in reps:
        st0 = 0
        st1 = 0
        st2 = 0
        st3 = 0
        st4 = 0
        st5 = 0
        for lineageTree_list in rep:
            for lineage_tree in lineageTree_list:
                for cell in lineage_tree.output_lineage:
                    if cell.state == 0:
                        st0 += 1
                    elif cell.state == 1:
                        st1 += 1
                    elif cell.state == 2:
                        st2 += 1
                    elif cell.state == 3:
                        st3 += 1
                    elif cell.state == 4:
                        st4 += 1
                    elif cell.state == 5:
                        st5 += 1
        s0.append(st0)
        s1.append(st1)
        s2.append(st2)
        s3.append(st3)
        s4.append(st4)
        s5.append(st5)

    return [s0, s1, s2, s3, s4, s5]


def reps_all_conditions(cn, one, two, three, tHMMobj_list):
    """collects all the states abundances for all replicates. Preparation for plotting."""
    indexes = [
        convertToIndex(cn),
        convertToIndex(one),
        convertToIndex(two),
        convertToIndex(three),
    ]
    reps = [separate_reps(indexes[i], tHMMobj_list[i]) for i in range(4)]

    abund = [state_abundance_perRep(reps[i]) for i in range(4)]
    return abund


def makeFigure():
    """Plot the bar charts of state abundances for all conditions and replicates."""

    ax, f = getSetup((7, 3), (1, 2))
    titles_L = ["control", "25 nM Lapatinib", "50 nM Lapatinib", "250 nM Lapatinib"]
    titles_G = ["control", "5 nM Gemcitabine", "10 nM Gemcitabine", "30 nM Gemcitabine"]
    labels_G = ["state 0", "state 1", "state 2", "state 3", "state 4"]
    labels_L = ["state 0", "state 1", "state 2", "state 3", "state 4", "state 5"]

    lpt = np.array(
        reps_all_conditions(
            lpt_cn_reps, lpt_25_reps, lpt_50_reps, lpt_250_reps, lapt_tHMMobj_list
        )
    )
    LPT = np.sum(lpt, axis=0)
    gem = np.array(
        reps_all_conditions(
            gem_cn_reps, gem_5_reps, gem_10_reps, gem_30_reps, gemc_tHMMobj_list
        )
    )
    GEM = np.sum(gem, axis=0)
    x1 = np.arange(len(labels_L))
    x2 = np.arange(len(labels_G))
    width = 0.2

    ax[0].bar(x1 - width, LPT[:, 0], width, label="rep1")
    ax[0].bar(x1, LPT[:, 1], width, label="rep2")
    ax[0].bar(x1 + width, LPT[:, 2], width, label="rep3")
    ax[0].set_title("Lapatinib")
    ax[0].set_xlabel("States")
    ax[0].set_ylabel("State # frequencies")
    ax[0].set_xticks(x1)
    ax[0].set_xticklabels(labels_L)
    ax[0].legend()

    ax[1].bar(x2 - width, GEM[0:5, 0], width, label="rep1")
    ax[1].bar(x2, GEM[0:5, 1], width, label="rep2")
    ax[1].bar(x2 + width, GEM[0:5, 2], width, label="rep3")
    ax[1].set_title("Gemcitabine")
    ax[1].set_xlabel("States")
    ax[1].set_ylabel("State # frequencies")
    ax[1].set_xticks(x2)
    ax[1].set_xticklabels(labels_G)
    ax[1].legend()

    f.tight_layout()

    return f
