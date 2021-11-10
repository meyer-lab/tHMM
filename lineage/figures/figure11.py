""" This file depicts the distribution of phase lengths versus the states for each concentration of lapatinib. """
import pickle
from string import ascii_lowercase

from .figureCommon import getSetup, plot_all
from ..plotTree import plot_networkx
from ..Lineage_collections import lpt_cn_reps, lpt_25_reps, lpt_50_reps, lpt_250_reps

concs = ["control", "lapatinib 25 nM", "lapatinib 50 nM", "lapatinib 250 nM"]
concsValues = ["control", "25 nM", "50 nM", "250 nM"]

pik1 = open("lapatinibs.pkl", "rb")
lapt_tHMMobj_list = []
for i in range(4):
    lapt_tHMMobj_list.append(pickle.load(pik1))

T_lap = lapt_tHMMobj_list[0].estimate.T
num_states = lapt_tHMMobj_list[0].num_states

cn_rep1 = [lapt_tHMMobj_list[0].X[0:lpt_cn_reps[0]], lapt_tHMMobj_list[0].X[lpt_cn_reps[0]:lpt_cn_reps[1]]]

def convertToIndex(lpt_cn_reps):
    for i, val in enumerate(lpt_cn_reps):
        if i == 0:
            lpt_cn_reps[i] = val
        else:
            lpt_cn_reps[i] = lpt_cn_reps[i] + lpt_cn_reps[i-1]
    return [0] + lpt_cn_reps
    
def separate_reps(i, rep_indx_list, thmm_list):
    """ For each condition, ie., control, 25nM, etc., 
    makes a list of lists containing lineage_trees of separate replicates."""
    resp = []
    for k, val in enumerate(rep_indx_list):
        reps.append([thmm_list[i].X[val:rep_indx_list[k+1]])
        if k >= 3:
            break
    assert len(reps) == 3
    return reps



def makeFigure():
    """ Makes figure 11. """

    ax, f = getSetup((17, 7.5), (2, 7))
    plot_all(ax, num_states, lapt_tHMMobj_list, "Laptinib", concs, concsValues)
    for i in range(3, 7):
        ax[i].set_title(concs[i - 3], fontsize=16)
        ax[i].text(-0.2, 1.25, ascii_lowercase[i - 2], transform=ax[i].transAxes, fontsize=16, fontweight="bold", va="top")
        ax[i].axis('off')

    return f

# plot_networkx(T_lap.shape[0], T_lap, 'lapatinib')
