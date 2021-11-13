""" This file includes functions to separate cell states of different replicates, and plot the barchart for each condition."""
import pickle
import numpy as np
from .figureCommon import getSetup
from lineage.Lineage_collections import lpt_cn_reps, lpt_25_reps, lpt_50_reps, lpt_250_reps, gem_cn_reps, gem_5_reps, gem_10_reps, gem_30_reps


pik1 = open("lapatinibs.pkl", "rb")
lapt_tHMMobj_list = []
for i in range(4):
    lapt_tHMMobj_list.append(pickle.load(pik1))

pik2 = open("gemcitabines.pkl", "rb")
gemc_tHMMobj_list = []
for i in range(4):
    gemc_tHMMobj_list.append(pickle.load(pik2))


def convertToIndex(lpt_cn_reps):
    """ Takes the list containing the lineage numbers of each replicate, 
    and returning the list of their index correspondance."""
    for i, val in enumerate(lpt_cn_reps):
        if i == 0:
            lpt_cn_reps[i] = val
        else:
            lpt_cn_reps[i] = lpt_cn_reps[i] + lpt_cn_reps[i-1]
    return [0] + lpt_cn_reps
    
def separate_reps(rep_indx_list, thmm):
    """ For a given condition, ie., control, 25nM, etc., 
    makes a list of lists containing lineage_trees of separate replicates."""
    reps = []
    for k, val in enumerate(rep_indx_list):
        reps.append([thmm.X[val:rep_indx_list[k+1]]])
        if k >= 2:
            break
    assert len(reps) == 3
    return reps

def state_abundance_perRep(reps):
    """Finds the number of cells in each state for all replicates of a condition. """
    s0 = []; s1 = []; s2 = []; s3 = []; s4 = []; s5 = []
    for rep in reps:
        st0 = 0; st1 = 0; st2 = 0; st3 = 0; st4 = 0; st5 = 0
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
    """ collects all the states abundances for all replicates. Preparation for plotting. """
    indexes = [convertToIndex(cn), convertToIndex(one), convertToIndex(two), convertToIndex(three)]
    reps = [separate_reps(indexes[i], tHMMobj_list[i]) for i in range(4)]

    abund = [state_abundance_perRep(reps[i]) for i in range(4)]
    return abund


def makeFigure():
    """Plot the bar charts of state abundances for all conditions and replicates. """

    ax, f = getSetup((17, 7.5), (2, 4))
    titles_L = ["control", "25 nM Lapatinib", "50 nM Lapatinib", "250 nM Lapatinib"]
    titles_G = ["control", "5 nM Gemcitabine", "10 nM Gemcitabine", "30 nM Gemcitabine"]
    labels_G = ["state 0", "state 1", "state 2", "state 3", "state 4"]
    labels_L = ["state 0", "state 1", "state 2", "state 3", "state 4", "state 5"]

    LPT = np.array(reps_all_conditions(lpt_cn_reps, lpt_25_reps, lpt_50_reps, lpt_250_reps, lapt_tHMMobj_list))
    GEM = np.array(reps_all_conditions(gem_cn_reps, gem_5_reps, gem_10_reps, gem_30_reps, gemc_tHMMobj_list))
    x1 = np.arange(len(labels_L))
    x2 = np.arange(len(labels_G))
    width = 0.2
    for i in range(4):
        ax[i].bar(x1 - width, LPT[i, :, 0], width, label="rep1")
        ax[i].bar(x1, LPT[i, :, 1], width, label="rep2")
        ax[i].bar(x1 + width, LPT[i, :, 2], width, label="rep3")
        ax[i].set_title(titles_L[i])
        ax[i].set_xlabel("States")
        ax[i].set_ylabel("State # frequencies")
        ax[i].set_xticks(x1)
        ax[i].set_xticklabels(labels_L)
        ax[i].legend()

        ax[i+4].bar(x2 - width, GEM[i, 0:5, 0], width, label="rep1")
        ax[i+4].bar(x2, GEM[i, 0:5, 1], width, label="rep2")
        ax[i+4].bar(x2 + width, GEM[i, 0:5, 2], width, label="rep3")
        ax[i+4].set_title(titles_G[i])
        ax[i+4].set_xlabel("States")
        ax[i+4].set_ylabel("State # frequencies")
        ax[i+4].set_xticks(x2)
        ax[i+4].set_xticklabels(labels_G)
        ax[i+4].legend()
    
        f.tight_layout()
    return f