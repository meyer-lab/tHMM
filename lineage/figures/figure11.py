""" This file depicts the distribution of phase lengths versus the states for each concentration of lapatinib. """
import pickle
from string import ascii_lowercase

from .figureCommon import getSetup, plot_all
from ..plotTree import plot_networkx
from ..Lineage_collections import lpt_cn_reps, lpt_25_reps, lpt_50_reps, lpt_250_reps, gem_cn_reps, gem_5_reps, gem_10_reps, gem_30_reps

concs = ["control", "lapatinib 25 nM", "lapatinib 50 nM", "lapatinib 250 nM"]
concsValues = ["control", "25 nM", "50 nM", "250 nM"]

pik1 = open("lapatinibs.pkl", "rb")
lapt_tHMMobj_list = []
for i in range(4):
    lapt_tHMMobj_list.append(pickle.load(pik1))

T_lap = lapt_tHMMobj_list[0].estimate.T
num_states = lapt_tHMMobj_list[0].num_states


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
    resp = []
    for k, val in enumerate(rep_indx_list):
        reps.append([thmm.X[val:rep_indx_list[k+1]])
        if k >= 3:
            break
    assert len(reps) == 3
    return reps

def state_abundance_perRep(reps):
    """Finds the number of cells in each state for all replicates of a condition. """
    s0 = []; s1 = []; s2 = []; s3 = []; s4 = []; s5 = []
    for rep in reps:
        st0 = 0; st1 = 0; st2 = 0; st3 = 0; st4 = 0; st5 = 0
        for lineageTree in rep:
            for cell in lineageTree.output_lineage:
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


def plot_barcharts(ss):
    """Plot the bar charts of state abundances for all conditions and replicates. """

    labels_L = ["control", "25 nM", "50 nM", "250 nM"]
    labels_G = ["control", "5 nM", "10 nM", "30 nM"]
    LPT = reps_all_conditions(lpt_cn_reps, lpt_25_reps, lpt_50_reps, lpt_250_reps, lapt_tHMMobj_list)
    GEM = reps_all_conditions(gem_cn_reps, gem_5_reps, gem_10_reps, gem_30_reps, gemc_tHMMobj_list)


def makeFigure():
    """ Makes figure 11. """

    ax, f = getSetup((17, 7.5), (2, 7))
    plot_all(ax, num_states, lapt_tHMMobj_list, "Laptinib", concs, concsValues)
    for i in range(3, 7):
        ax[i].set_title(concs[i - 3], fontsize=16)
        ax[i].text(-0.2, 1.25, ascii_lowercase[i - 2], transform=ax[i].transAxes, fontsize=16, fontweight="bold", va="top")
        ax[i].axis('off')

    return f
