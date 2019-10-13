"""
This creates Figure 9 which plots with the following characteristics both pruning, 2states, state assignment accuracy.
"""
from .figureCommon import subplotLabel, getSetup
from matplotlib.ticker import MaxNLocator
from ..Analyze import accuracy, accuracyG, Analyze
from ..LineageTree import LineageTree
from ..StateDistribution import StateDistribution
from ..StateDistribution2 import StateDistribution2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rc('font', **{'family': 'sans-serif', 'size': 25})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
plt.rc('text', usetex=True)
plt.rc('xtick', **{'labelsize':'medium'})
plt.rc('ytick', **{'labelsize':'medium'})

def makeFigure():
    """ makes figure 9 """

    # Get list of axis objects
    ax, f = getSetup((24, 6), (1, 3))
    x, accuracies, tr, pi = accuracy_increased_cells()
    figure_maker(ax, x, accuracies, tr, pi)
    
    return f


def accuracy_increased_cells():
    """ Calculates accuracy and parameter estimation by increasing the number of cells in a lineage for a two-state model. """

    # pi: the initial probability vector
    piiii = np.array([0.6, 0.4], dtype="float")

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
    
    desired_num_cells = 2**12 - 1
    experiment_time = 100
    num_lineages = list(range(1, 25))
    list_of_lineages = []

    for num in num_lineages:
        X1 = []
        for lineages in range(num):
            # Creating an unpruned and pruned lineage
            lineage = LineageTree(piiii, T, E, desired_num_cells, experiment_time, prune_condition='both', prune_boolean=True)
            while len(lineage.output_lineage) < 16:
                lineage = LineageTree(piiii, T, E, (2**12)-1, experiment_time, prune_condition='both', prune_boolean=True)

            # Setting then into a list or a population of lineages and collecting the length of each lineage
            X1.append(lineage)
        # Adding populations into a holder for analysing
        list_of_lineages.append(X1)

    x = []
    accuracies = []
    tr = []
    pi = []

    for idx, X1 in enumerate(list_of_lineages):
        # Analyzing the lineages
        deltas, _, all_states, tHMMobj, _, _ = Analyze(X1, 2)
        
        # Collecting how many lineages are in each analysis
        x.append(len(X1))

        # Collecting the accuracies of the lineages
        acc1 = accuracy(tHMMobj, all_states)[0]*100
        while acc1 < 50:
            # Analyzing the lineages
            deltas, _, all_states, tHMMobj, _, _ = Analyze(X1, 2)

            # Collecting the accuracies of the lineages
            acc1 = accuracy(tHMMobj, all_states)[0]*100
        accuracies.append(acc1)
        
        # Transition and Pi estimates
        transition_mat = tHMMobj.estimate.T  # unpruned

        temp1 = T - transition_mat
        tr.append(np.linalg.norm(temp1))

        pi_mat = tHMMobj.estimate.pi
        t1 = piiii - pi_mat
        print(piiii, pi_mat, t1)
        pi.append(np.linalg.norm(t1))

    return x, accuracies, tr, pi


def moving_average(a, n=5):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def figure_maker(ax, x, accuracies, tr, pi):
    
    x_vs_acc = np.column_stack((x, accuracies))
    sorted_x_vs_acc = x_vs_acc[np.argsort(x_vs_acc[:, 0])]
    
    x_vs_tr = np.column_stack((x, tr))
    sorted_x_vs_tr = x_vs_tr[np.argsort(x_vs_tr[:, 0])]
    
    x_vs_pi = np.column_stack((x, pi))
    sorted_x_vs_pi = x_vs_pi[np.argsort(x_vs_pi[:, 0])]
    
    i = 0
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x)))))
    ax[i].set_xlabel('Number of Cells')
    ax[i].set_ylim(0, 110)
    ax[i].scatter(x, accuracies, c='k', marker="o", label='Accuracy', edgecolors='k', alpha=0.25)
    ax[i].plot(sorted_x_vs_acc[:, 0][4:], moving_average(sorted_x_vs_acc[:, 1]), c='k', label='Moving Average')
    ax[i].set_ylabel(r'Accuracy [\%]')
    ax[i].axhline(y=100, linestyle='--', linewidth=2, color='k', alpha=1) 
    ax[i].set_title('State Assignment Accuracy')
    ax[i].grid(linestyle='--')
    ax[i].tick_params(axis='both', which='major', grid_alpha=0.25)

    i += 1
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x)))))
    ax[i].set_xlabel('Number of Lineages')
    ax[i].scatter(x, tr, c='k', marker="o", edgecolors='k', alpha=0.25)
    ax[i].plot(sorted_x_vs_tr[:, 0][4:], moving_average(sorted_x_vs_tr[:, 1]), c='k', label='Moving Average')
    ax[i].set_ylabel(r'$||T-T_{est}||_{F}$')
    ax[i].axhline(y=0, linestyle='--', linewidth=2, color='k', alpha=1)
    ax[i].set_title('Transition Matrix Estimation')
    ax[i].grid(linestyle='--')
    ax[i].tick_params(axis='both', which='major', grid_alpha=0.25)
    
    i += 1
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x)))))
    ax[i].set_xlabel('Number of Lineages')
    ax[i].scatter(x, pi, c='k', marker="o", edgecolors='k', alpha=0.25)
    ax[i].plot(sorted_x_vs_pi[:, 0][4:], moving_average(sorted_x_vs_pi[:, 1]), c='k', label='Moving Average')
    ax[i].set_ylabel(r'$||\pi-\pi_{est}||_{2}$')
    ax[i].axhline(y=0, linestyle='--', linewidth=2, color='k', alpha=1)
    ax[i].set_title(r'Initial Seeding Density Estimation')
    ax[i].grid(linestyle='--')
    ax[i].tick_params(axis='both', which='major', grid_alpha=0.25)

