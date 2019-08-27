"""
This creates Figure 7. AIC Figure.
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
    
    num_lineages, accuracies_unpruned, bern_unpruned, exp_unpruned = accuracy_increased_lineages()
    figure_maker(ax[0:3],  num_lineages, accuracies_unpruned, bern_unpruned, exp_unpruned)
    
    
    f.tight_layout()
    return f

    
##-------------------- Figure 7   
def accuracy_increased_lineages():
    """ Calclates accuracy and parameter estimation by increasing the number of lineages. """
    # pi: the initial probability vector
    pi = np.array([0.5, 0.5], dtype="float")

    # T: transition probability matrix
    T = np.array([[0.99, 0.01],
                  [0.15, 0.85]], dtype='float')

    # State 0 parameters "Resistant"
    state0 = 0
    bern_p0 = 0.95
    exp_0 = 20.0

    # State 1 parameters "Susciptible"
    state1 = 1
    bern_p1 = 0.8
    exp_1 = 80.0

    state_obj0 = StateDistribution(state0, bern_p0, exp_0)
    state_obj1 = StateDistribution(state1, bern_p1, exp_1)

    E = [state_obj0, state_obj1]

    desired_num_cells = 2**4 - 1
    # increasing number of lineages from 1 to 10 and calculating accuracy and estimate parameters for both pruned and unpruned lineages.
    num_lineages = list(range(1, 10))

    accuracies_unpruned = []
    bern_unpruned = []
    exp_unpruned = []


    X_p = []
    for num in num_lineages:
        lineage_unpruned = LineageTree(pi, T, E, desired_num_cells, prune_boolean=False)


        X_p.append(lineage_unpruned)
        deltas, state_ptrs, all_states, tHMMobj, NF, LL = Analyze(X_p, 2) 
        acc1 = accuracy(X_p, all_states)
        accuracies_unpruned.append(100*acc1)        


        bern_p_total = []
        exp_total = []

        for state in range(tHMMobj.numStates):
            bern_p_estimate = tHMMobj.estimate.E[state].bern_p
            exp_estimate = tHMMobj.estimate.E[state].exp_scale_beta

            bern_p_total.append(bern_p_estimate)
            exp_total.append(exp_estimate)
            
        bern_unpruned.append(bern_p_total)
        exp_unpruned.append(exp_total)

        
    return num_lineages, accuracies_unpruned, bern_unpruned, exp_unpruned


def figure_maker(ax, num_lineages, accuracies_unpruned, bern_unpruned, exp_unpruned):
    x = num_lineages
    font=11
    font2 = 10
    ax[0].set_xlim((0, int(np.ceil(1.1 * max(x)))))
    ax[0].set_xlabel('Number of Lineages', fontsize=font2)
    ax[0].set_ylim(0, 110)
    ax[0].scatter(x, accuracies_unpruned, c='k', marker="o", label='Accuracy', alpha=0.3)
    ax[0].axhline(y=100, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='k', alpha=0.6)  # linestyle is dashdotdotted
    ax[0].set_ylabel(r'Accuracy (\%)', rotation=90, fontsize=font2)
    ax[0].get_yticks()
    ax[0].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[0].set_title('State Assignment Accuracy', fontsize=font)
    
    res = [[ i for i, j in bern_unpruned ], [ j for i, j in bern_unpruned ]] 
    ax[1].set_xlim((0, int(np.ceil(1.1 * max(x)))))
    ax[1].set_xlabel('Number of Lineages', fontsize=font2)
    ax[1].scatter(x, res[0], c='b', marker="o", label='Susceptible Unpruned', alpha=0.5)
    ax[1].scatter(x, res[1], c='r', marker="o", label='Resistant Unpruned', alpha=0.5)
    ax[1].set_ylabel('Bern $p$', rotation=90, fontsize=font2)
    ax[1].axhline(y=0.95, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b', alpha=0.6)
    ax[1].axhline(y=0.85, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[1].set_title('Bernoulli', fontsize=font)
    ax[1].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[1].legend(loc='best', framealpha=0.3)
    
    res = [[ i for i, j in exp_unpruned ], [ j for i, j in exp_unpruned ]] 
    ax[2].set_xlim((0, int(np.ceil(1.1 * max(x)))))
    ax[2].set_xlabel('Number of Lineages', fontsize=font2)
    ax[2].scatter(x, res[0], c='b', marker="o", label='Susceptible Unpruned', alpha=0.5)
    ax[2].scatter(x, res[1], c='r', marker="o", label='Resistant Unpruned', alpha=0.5)
    ax[2].set_ylabel(r'Exponential scale $\beta$', rotation=90, fontsize=font2)
    ax[2].axhline(y=20, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b', alpha=0.6)
    ax[2].axhline(y=80, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[2].set_title('Exponential', fontsize=font)
    ax[2].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[2].legend(loc='best', framealpha=0.3)