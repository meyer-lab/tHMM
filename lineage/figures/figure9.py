"""
This creates Figure 7 which plots with the following characteristics both pruning, 2states, state assignment accuracy.
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
    x, accuracies, tr, pi = accuracy_increased_cells()
    figure_maker(ax, x, accuracies, tr, pi)
    
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
    gamma_a1 = 10
    gamma_scale1 = 1

    state_obj0 = StateDistribution(state0, bern_p0, gamma_a0, gamma_loc, gamma_scale0)
    state_obj1 = StateDistribution(state1, bern_p1, gamma_a1, gamma_loc, gamma_scale1)
    E = [state_obj0, state_obj1]
    
    desired_num_cells = 2**11 - 1
    experiment_time = 500
    num_lineages = list(range(1, 10))
    list_of_lineages_unpruned = []


    for num in num_lineages:
        X1 = []
        for lineages in range(num):
            # Creating an unpruned and pruned lineage
            lineage_unpruned = LineageTree(piiii, T, E, desired_num_cells, experiment_time, prune_condition='both', prune_boolean=True)

            # Setting then into a list or a population of lineages and collecting the length of each lineage
            X1.append(lineage_unpruned)
        # Adding populations into a holder for analysing
        list_of_lineages_unpruned.append(X1)

    x = []
    accuracies = []
    tr = []
    pi = []

    for idx, X1 in enumerate(list_of_lineages_unpruned):
        # Analyzing the lineages
        deltas, _, all_states, tHMMobj, _, _ = Analyze(X1, 2)
        
        # Collecting how many lineages are in each analysis
        x.append(len(X1))

        # Collecting the accuracies of the lineages
        acc1 = accuracy(tHMMobj, all_states)[0]
        accuracies.append(acc1)

    # Transition and Pi estimates
        transition_mat = tHMMobj.estimate.T  # unpruned

        temp1 = T - transition_mat
        tr.append(np.linalg.norm(temp1))

        pi_mat = tHMMobj.estimate.pi
        t1 = piiii - pi_mat
        pi.append(np.linalg.norm(t1))

    return x, accuracies, tr, pi


def figure_maker(ax, x, accuracies, tr, pi):

    font = 11
    font2 = 10
    i = 0
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x)))))
    ax[i].set_xlabel('Number of Lineages', fontsize=font2)
    ax[i].set_ylim(0, 110)
    ax[i].scatter(x, accuracies, c='k', marker="o", label='Accuracy', alpha=0.3)
    ax[i].axhline(y=100, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='k', alpha=0.6)  # linestyle is dashdotdotted
    ax[i].set_ylabel(r'Accuracy (\%)', rotation=90, fontsize=font2)
    ax[i].get_yticks()
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].set_title('State Assignment Accuracy', fontsize=font)

    i += 1
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x)))))
    ax[i].set_xlabel('Number of Lineages', fontsize=font2)
    ax[i].scatter(x, tr, c='k', marker="o", alpha=0.5)
    ax[i].set_ylabel(r'$||T-T_{est}||_{F}$', rotation=90, fontsize=font2)
    ax[i].axhline(y=0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='k', alpha=0.6)
    ax[i].set_title('Norm Transition', fontsize=font)
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)

    i += 1
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x)))))
    ax[i].set_xlabel('Number of Lineages', fontsize=font2)
    ax[i].scatter(x, pi, c='k', marker="o", alpha=0.5)
    ax[i].set_ylabel(r'$||\pi-\pi_{est}||_{2}$', rotation=90, fontsize=font2)
    ax[i].axhline(y=0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='k', alpha=0.6)
    ax[i].set_title('Norm $\pi$', fontsize=font)
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)

