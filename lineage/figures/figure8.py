"""
This creates Figure 8.
"""
from .figureCommon import subplotLabel, getSetup
from matplotlib.ticker import MaxNLocator
from ..Analyze import accuracy, accuracyG, Analyze
from ..LineageTree import LineageTree
from ..StateDistribution import StateDistribution
from ..StateDistribution2 import StateDistribution2

import numpy as np
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

def makeFigure():
    """ makes figure 4 """

    # Get list of axis objects
    ax, f = getSetup((12, 4), (1, 3))
    x, bern_unpruned, bern_p0, bern_p1, gamma_a_unpruned, gamma_a0, gamma_a1, gamma_scale_unpruned, gamma_scale0, gamma_scale1 = accuracy_increased_cells()
    figure_maker(ax, x, bern_unpruned, bern_p0, bern_p1, gamma_a_unpruned, gamma_a0, gamma_a1, gamma_scale_unpruned, gamma_scale0, gamma_scale1)
    
    return f


def accuracy_increased_cells():
    """ Calculates accuracy and parameter estimation by increasing the number of cells in a lineage for a two-state model. """

    # pi: the initial probability vector
    piiii = np.array([0.15, 0.85], dtype="float")

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
    gamma_a1 = 11
    gamma_scale1 = 1.1

    state_obj0 = StateDistribution(state0, bern_p0, gamma_a0, gamma_loc, gamma_scale0)
    state_obj1 = StateDistribution(state1, bern_p1, gamma_a1, gamma_loc, gamma_scale1)
    E = [state_obj0, state_obj1]

    desired_num_cells = np.logspace(5, 10, num=20, base=2.0)
    desired_num_cells = [num_cell - 1 for num_cell in desired_num_cells]

    x = []
    accuracies_unpruned = []
    bern_unpruned = []  
    gamma_a_unpruned = []
    gamma_scale_unpruned = []
    
    
    times = np.linspace(100, 1000, 10)

    for experiment_time in times:
        # Creating an unpruned and pruned lineage
        lineage = LineageTree(piiii, T, E, (2**12)-1, experiment_time, prune_condition='both', prune_boolean=True)

        # Setting then into a list or a population of lineages and collecting the length of each lineage
        X1 = [lineage]
        x.append(len(lineage.output_lineage))

        # Analyzing the lineages
        deltas, _, all_states, tHMMobj, _, _ = Analyze(X1, 2)

        # Collecting the accuracies of the lineages
        acc1 = accuracy(tHMMobj, all_states)[0]
        accuracies_unpruned.append(acc1)

        # Collecting the parameter estimations
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

        
    return x, bern_unpruned, bern_p0, bern_p1, gamma_a_unpruned, gamma_a0, gamma_a1, gamma_scale_unpruned, gamma_scale0, gamma_scale1


def figure_maker(ax, x, bern_unpruned, bern_p0, bern_p1, gamma_a_unpruned, gamma_a0, gamma_a1, gamma_scale_unpruned, gamma_scale0, gamma_scale1):

    font = 11
    font2 = 10
    i = 0
    res = [[i for i, j in bern_unpruned], [j for i, j in bern_unpruned]]
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)
    ax[i].scatter(x, res[0], c='b', marker="o", alpha=0.5)
    ax[i].scatter(x, res[1], c='r', marker="o", alpha=0.5)   
    ax[i].set_ylabel('Bern $p$', rotation=90, fontsize=font2)
    ax[i].axhline(y=bern_p0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, label = 'resistant', color='b', alpha=0.6)
    ax[i].axhline(y=bern_p1, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[i].set_title('Bernoulli', fontsize=font)
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)

    i += 1
    res = [[i for i, j in gamma_a_unpruned], [j for i, j in gamma_a_unpruned]]
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)
    ax[i].scatter(x, res[0], c='b', marker="o", alpha=0.5)
    ax[i].scatter(x, res[1], c='r', marker="o", alpha=0.5)
    ax[i].set_ylabel(r'Gamma a $\beta$', rotation=90, fontsize=font2)
    ax[i].axhline(y=gamma_a0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, label = 'resistant', color='b', alpha=0.6)
    ax[i].axhline(y=gamma_a1, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[i].set_title('Gamma', fontsize=font)
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)

    i += 1
    res = [[i for i, j in gamma_scale_unpruned], [j for i, j in gamma_scale_unpruned]]
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)
    ax[i].scatter(x, res[0], c='b', marker="o", alpha=0.5)
    ax[i].scatter(x, res[1], c='r', marker="o", alpha=0.5)
    ax[i].set_ylabel(r'Gamma scale $\alpha$', rotation=90, fontsize=font2)
    ax[i].axhline(y=gamma_scale0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2,label = 'resistant', color='b', alpha=0.6)
    ax[i].axhline(y=gamma_scale1, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[i].set_title('Gamma', fontsize=font)
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)

    
    
    
    
