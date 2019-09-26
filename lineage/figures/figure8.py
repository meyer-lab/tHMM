"""
This creates Figure 8.
"""
from .figureCommon import subplotLabel, getSetup
from matplotlib.ticker import MaxNLocator
from ..Analyze import accuracy, accuracyG, Analyze, getAIC
from ..LineageTree import LineageTree
from ..StateDistribution import StateDistribution
from ..StateDistribution2 import StateDistribution2

import numpy as np
import copy as cp
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


def makeFigure():
    """ Main figure generating function for Fig. 6 """
    ax, f = getSetup((5, 10), (2, 1))

    desred_num_states, AIC_unpruned, AIC_pruned = AIC_increased_cells()
    figure_maker(ax, desred_num_states, AIC_unpruned, AIC_pruned)

    return f



def AIC_increased_cells():
    """ Calculates accuracy and parameter estimation by increasing the number of cells in a lineage for a two-state model. """

    # pi: the initial probability vector
    pi = np.array([0.6, 0.4], dtype="float")

    # T: transition probability matrix
    T = np.array([[0.85, 0.15],
                  [0.15, 0.85]], dtype="float")

    # State 0 parameters "Resistant"
    state0 = 0
    bern_p0 = 0.99
    gamma_a0 = 20
    gamma_loc = 0
    gamma_scale0 = 5

    # State 1 parameters "Susceptible"
    state1 = 1
    bern_p1 = 0.88
    gamma_a1 = 10
    gamma_scale1 = 1

    state_obj0 = StateDistribution(state0, bern_p0, gamma_a0, gamma_loc, gamma_scale0)
    state_obj1 = StateDistribution(state1, bern_p1, gamma_a1, gamma_loc, gamma_scale1)
    E = [state_obj0, state_obj1]
    
    
    desred_num_states = [1, 2, 3, 4]
    
    desired_num_cells = np.logspace(5, 10, num=10, base=2.0)
    desired_num_cells = [num_cell - 1 for num_cell in desired_num_cells]

    AIC_unpruned = np.zeros(shape=(len(desired_num_cells),len(desred_num_states)))
    AIC_pruned =  np.zeros(shape=(len(desired_num_cells),len(desred_num_states)))

    for num_cells_idx, num_cells in enumerate(desired_num_cells):
        for num_states in desred_num_states:
            # Creating an unpruned and pruned lineage
            lineage_unpruned = LineageTree(pi, T, E, num_cells, prune_boolean=False)
            # if the length of the pruned lineage tree is less than 5 cells, don't analyze either the pruned
            # or the unpruned lineage and skip
            if lineage_unpruned.__len__(True) <= 5:
                continue
            lineage_pruned = cp.deepcopy(lineage_unpruned)
            lineage_pruned.prune_boolean = True

            # Setting then into a list or a population of lineages and collecting the length of each lineage
            X1 = [lineage_unpruned]
            X2 = [lineage_pruned]
            # Analyzing the lineages
            deltas, _, all_states, tHMMobj, _, LL = Analyze(X1, num_states)
            deltas2, _, all_states2, tHMMobj2, _, LL2 = Analyze(X2, num_states)
            
            # AIC
            AIC_ls, LL_ls, AIC_degrees_of_freedom = getAIC(tHMMobj, LL)
            AIC_ls2, LL_ls2, AIC_degrees_of_freedom2 = getAIC(tHMMobj2, LL2)
            AIC_unpruned[num_cells_idx, num_states-1] = (np.mean(AIC_ls))
            AIC_pruned[num_cells_idx, num_states-1] = (np.mean(AIC_ls2))

    return desred_num_states, AIC_unpruned, AIC_pruned


def figure_maker(ax, desred_num_states, AIC_unpruned, AIC_pruned):

    font = 11
    font2 = 10
    i = 0
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(desred_num_states)))))
    ax[i].set_xlabel('Number of States', fontsize=font2)
    ax[i].boxplot(AIC_unpruned)
    ax[i].set_ylabel(r'AIC', rotation=90, fontsize=font2)
    ax[i].get_yticks()
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].set_title('State Assignment AIC', fontsize=font)

    i += 1
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(desred_num_states)))))
    ax[i].set_xlabel('Number of States', fontsize=font2)
    ax[i].boxplot(AIC_pruned)
    ax[i].set_ylabel(r'AIC', rotation=90, fontsize=font2)
    ax[i].get_yticks()
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].set_title('State Assignment AIC', fontsize=font)
    
    