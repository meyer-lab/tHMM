"""
This creates Figure 6.
"""
from .figureCommon import getSetup
from ..Analyze import accuracy, Analyze
from ..LineageTree import LineageTree
from ..StateDistribution import StateDistribution

import numpy as np
import copy as cp

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

def makeFigure():
    """ Main figure generating function for Fig. 6 """
    ax, f = getSetup((15, 5), (1, 3))
    
    desired_num_cells, accuracies_unpruned, bern_unpruned, bern_p0, bern_p1, gamma_a_unpruned, gamma_a0, gamma_a1, gamma_scale_unpruned, gamma_scale0, gamma_scale1 = accuracy_increased_cells()
    figure_maker(ax[0:3], desired_num_cells, accuracies_unpruned, bern_unpruned, bern_p0, bern_p1, gamma_a_unpruned, gamma_a0, gamma_a1, gamma_scale_unpruned, gamma_scale0, gamma_scale1)
    
    
    f.tight_layout()
    return f




#-------------------- Figure 6 
def accuracy_increased_cells():
    """ Calclates accuracy and parameter estimation by increasing the number of cells in a lineage for a two-state model. """

    # pi: the initial probability vector
    pi = np.array([0.6, 0.4], dtype="float")

    # T: transition probability matrix
    T = np.array([[0.85, 0.15],
                  [0.15, 0.85]], dtype="float")

    # State 0 parameters "Resistant"
    state0 = 0
    bern_p0 = 0.99
    gamma_a0 = 20
    gamma_scale0 = 5

    # State 1 parameters "Susceptible"
    state1 = 1
    bern_p1 = 0.88
    gamma_a1 = 10
    gamma_scale1 = 1

    state_obj0 = StateDistribution(state0, bern_p0, gamma_a0, gamma_scale0)
    state_obj1 = StateDistribution(state1, bern_p1, gamma_a1, gamma_scale1)

    E = [state_obj0, state_obj1]
    # the key part in this function
    desired_num_cells = np.logspace(7, 10, num=50, base=2.0)
    desired_num_cells = [num_cell-1 for num_cell in desired_num_cells]

    accuracies_unpruned = []
    #accuracies_pruned = []
    bern_unpruned = []
    gamma_a_unpruned = []
    gamma_scale_unpruned = []

    for num in desired_num_cells:
        print(num)
        lineage_unpruned = LineageTree(pi, T, E, num, prune_boolean=False)


        X1 = [lineage_unpruned]
        print("unpruned")
        deltas, state_ptrs, all_states, tHMMobj, NF, LL = Analyze(X1, 2) 
        acc1 = accuracy(X1, all_states)
        accuracies_unpruned.append(100*acc1)        


        bern_p_total = ()
        gamma_a_total = ()
        gamma_scale_total = ()
        for state in range(tHMMobj.numStates):
            bern_p_total += (tHMMobj.estimate.E[state].bern_p,)
            gamma_a_total += (tHMMobj.estimate.E[state].gamma_a,)
            gamma_scale_total += (tHMMobj.estimate.E[state].gamma_scale,)
            

        bern_unpruned.append(bern_p_total)
        gamma_a_unpruned.append(gamma_a_total)
        gamma_scale_unpruned.append(gamma_scale_total)

        
    return desired_num_cells, accuracies_unpruned, bern_unpruned, bern_p0, bern_p1, gamma_a_unpruned, gamma_a0, gamma_a1, gamma_scale_unpruned, gamma_scale0, gamma_scale1


def figure_maker(ax, desired_num_cells, accuracies_unpruned, bern_unpruned, bern_p0, bern_p1, gamma_a_unpruned, gamma_a0, gamma_a1, gamma_scale_unpruned, gamma_scale0, gamma_scale1):
    x = desired_num_cells
    font=11
    font2 = 10
    ax[0].set_xlim((0, int(np.ceil(1.1 * max(x)))))
    ax[0].set_xlabel('Number of Cells', fontsize=font2)
    ax[0].set_ylim(0, 110)
    ax[0].scatter(x, accuracies_unpruned, c='k', marker="o", label='Accuracy', alpha=0.3)
    ax[0].axhline(y=100, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='k', alpha=0.6)  # linestyle is dashdotdotted
    ax[0].set_ylabel(r'Accuracy (\%)', rotation=90, fontsize=font2)
    ax[0].get_yticks()
    ax[0].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[0].set_title('State Assignment Accuracy', fontsize=font)
    
    res = [[ i for i, j in bern_unpruned ], [ j for i, j in bern_unpruned ]] 
    ax[1].set_xlim((0, int(np.ceil(1.1 * max(x)))))
    ax[1].set_xlabel('Number of Cells', fontsize=font2)
    ax[1].scatter(x, res[0], c='b', marker="o", label='Susceptible Unpruned', alpha=0.5)
    ax[1].scatter(x, res[1], c='r', marker="o", label='Resistant Unpruned', alpha=0.5)
    ax[1].set_ylabel('Bern $p$', rotation=90, fontsize=font2)
    ax[1].axhline(y=bern_p0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b', alpha=0.6)
    ax[1].axhline(y=bern_p1, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[1].set_title('Bernoulli', fontsize=font)
    ax[1].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[1].legend(loc='best', framealpha=0.3)
    
    res = [[ i for i, j in gamma_a_unpruned ], [ j for i, j in gamma_a_unpruned ]] 
    ax[2].set_xlim((0, int(np.ceil(1.1 * max(x)))))
    ax[2].set_xlabel('Number of Cells', fontsize=font2)
    ax[2].scatter(x, res[0], c='b', marker="o", label='Susceptible Unpruned', alpha=0.5)
    ax[2].scatter(x, res[1], c='r', marker="o", label='Resistant Unpruned', alpha=0.5)
    ax[2].set_ylabel(r'Gamma a $\beta$', rotation=90, fontsize=font2)
    ax[2].axhline(y=gamma_a0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b', alpha=0.6)
    ax[2].axhline(y=gamma_a1, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[2].set_title('Gamma', fontsize=font)
    ax[2].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[2].legend(loc='best', framealpha=0.3)
        
        
        

    
    
