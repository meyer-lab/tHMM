"""
File: figure3.py
Purpose: Generates figure 3.
Figure 3 analyzes heterogeneous (2 state), pruned (by both time and fate), single lineages
(no more than one lineage per population) with at least 16 cells over increasing experimental
times.
"""
import numpy as np
from matplotlib import gridspec, pyplot as plt

from .figureCommon import moving_average
from ..Analyze import get_results, run_Analyze_over
from ..LineageTree import LineageTree
from ..StateDistribution import StateDistribution


def getSetup(figsize):
    """Setup figures."""

    plt.rc('font', **{'family': 'sans-serif', 'size': 25})
    # for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})
    plt.rc('text', usetex=True)
    plt.rc('xtick', **{'labelsize': 'medium'})
    plt.rc('ytick', **{'labelsize': 'medium'})

    # Setup plotting space
    f = plt.figure(figsize=figsize)

    # Make grid
    gs1 = gridspec.GridSpec(2, 6, figure=f)

    # Get list of axis objects
    ax = [f.add_subplot(gs1[0, 0:2]), f.add_subplot(gs1[0, 2:4]),
          f.add_subplot(gs1[0, 4:6]), f.add_subplot(gs1[-1, 1:3]),
          f.add_subplot(gs1[-1, 3:5])]

    return (ax, f)


def makeFigure():
    """
    Makes figures 7 & 8.
    """

    # Get list of axis objects
    ax, f = getSetup((21, 12))
#     f.subplot2grid(shape, loc, rowspan=1, colspan=1)
    accuracy_increased_cells()
    #figure_maker(ax, x, accuracies, tr, pi, bern_pruned, bern_p0, bern_p1, gamma_a_pruned, gamma_a0, gamma_a1, gamma_scale_pruned, gamma_scale0, gamma_scale1)
    f.tight_layout()

    return f


def accuracy_increased_cells():
    """
    Calculates accuracy and transition rate estimation over an increasing number of cells in a lineage for an pruned two-state model.
    """

    # pi: the initial probability vector
    piiii = np.array([0.6, 0.4], dtype="float")

    # T: transition probability matrix
    T = np.array([[0.75, 0.25],
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

    # Creating a list of populations to analyze over
    times = np.linspace(100, 1000, 25)
    list_of_populations = []
    for experiment_time in times:
        # Creating an unpruned and pruned lineage
        lineage = LineageTree(piiii, T, E, (2**12) - 1, experiment_time, prune_condition='both', prune_boolean=True)
        
        while len(lineage.output_lineage) < 16:
            del lineage
            lineage = LineageTree(piiii, T, E, (2**12) - 1, experiment_time, prune_condition='both', prune_boolean=True)

        # Adding populations into a holder for analysing
        list_of_populations.append([lineage])

        
    # TODO: Analyzing the lineages in the list of populations (parallelized function)

    # TODO: Collecting the results of analyzing the lineages 


    return 

def figure_maker():
    """
    Makes figure 3.
    """
    i = 0
    res = [[i for i, j in bern_pruned], [j for i, j in bern_pruned]]
    ax[i].set_xlim((16, int(np.ceil(4 * max(x)))))
    ax[i].set_xlabel('Number of Cells')
    ax[i].scatter(x, res[0], c='#F9Cb9C', edgecolors='k', marker="o", alpha=0.5)
    ax[i].scatter(x, res[1], c='#A4C2F4', edgecolors='k', marker="o", alpha=0.5)
    ax[i].set_ylabel('Bernoulli $p$')
    ax[i].set_ylim([0.85, 1.01])
    ax[i].axhline(y=bern_p0, linestyle='--', linewidth=2, label='Resistant', color='#F9Cb9C', alpha=1)
    ax[i].axhline(y=bern_p1, linestyle='--', linewidth=2, label='Susceptible', color='#A4C2F4', alpha=1)
    ax[i].set_title(r'Bernoulli $p$')
    ax[i].grid(linestyle='--')
    ax[i].set_xscale('log', basex=2)
    ax[i].tick_params(axis='both', which='major', grid_alpha=0.25)

    i += 1
    res = [[i for i, j in gamma_a_pruned], [j for i, j in gamma_a_pruned]]
    ax[i].set_xlim((16, int(np.ceil(4 * max(x)))))
    ax[i].set_xlabel('Number of Cells')
    ax[i].scatter(x, res[0], c='#F9Cb9C', edgecolors='k', marker="o", alpha=0.5)
    ax[i].scatter(x, res[1], c='#A4C2F4', edgecolors='k', marker="o", alpha=0.5)
    ax[i].set_ylabel(r'Gamma $k$')
    ax[i].set_ylim([5, 25])
    ax[i].axhline(y=gamma_a0, linestyle='--', linewidth=2, label='Resistant', color='#F9Cb9C', alpha=1)
    ax[i].axhline(y=gamma_a1, linestyle='--', linewidth=2, label='Susceptible', color='#A4C2F4', alpha=1)
    ax[i].set_title(r'Gamma $k$')
    ax[i].grid(linestyle='--')
    ax[i].set_xscale('log', basex=2)
    ax[i].tick_params(axis='both', which='major', grid_alpha=0.25)

    i += 1
    res = [[i for i, j in gamma_scale_pruned], [j for i, j in gamma_scale_pruned]]
    ax[i].set_xlim((16, int(np.ceil(4 * max(x)))))
    ax[i].set_xlabel('Number of Cells')
    ax[i].scatter(x, res[0], c='#F9Cb9C', edgecolors='k', marker="o", alpha=0.5)
    ax[i].scatter(x, res[1], c='#A4C2F4', edgecolors='k', marker="o", alpha=0.5)
    ax[i].set_ylabel(r'Gamma $\theta$')
    ax[i].set_ylim([0, 7])
    ax[i].axhline(y=gamma_scale0, linestyle='--', linewidth=2, label='Resistant', color='#F9Cb9C', alpha=1)
    ax[i].axhline(y=gamma_scale1, linestyle='--', linewidth=2, label='Susceptible', color='#A4C2F4', alpha=1)
    ax[i].set_title(r'Gamma $\theta$')
    ax[i].grid(linestyle='--')
    ax[i].set_xscale('log', basex=2)
    ax[i].tick_params(axis='both', which='major', grid_alpha=0.25)
    ax[i].legend()

    x_vs_acc = np.column_stack((x, accuracies))
    sorted_x_vs_acc = x_vs_acc[np.argsort(x_vs_acc[:, 0])]

    x_vs_tr = np.column_stack((x, tr))
    sorted_x_vs_tr = x_vs_tr[np.argsort(x_vs_tr[:, 0])]

    i += 1
    ax[i].set_xlim((16, int(np.ceil(4 * max(x)))))
    ax[i].set_xlabel('Number of Cells')
    ax[i].set_ylim(0, 110)
    ax[i].scatter(x, accuracies, c='k', marker="o", label='Accuracy', edgecolors='k', alpha=0.25)
    ax[i].plot(sorted_x_vs_acc[:, 0][49:], moving_average(sorted_x_vs_acc[:, 1]), c='k', label='Moving Average')
    ax[i].set_xscale('log', basex=2)
    ax[i].set_ylabel(r'Accuracy [\%]')
    ax[i].axhline(y=100, linestyle='--', linewidth=2, color='k', alpha=1)
    ax[i].set_title('State Assignment Accuracy')
    ax[i].grid(linestyle='--')
    ax[i].tick_params(axis='both', which='major', grid_alpha=0.25)

    i += 1
    ax[i].set_xlim((16, int(np.ceil(4 * max(x)))))
    ax[i].set_xlabel('Number of Cells')
    ax[i].scatter(x, tr, c='k', marker="o", edgecolors='k', alpha=0.25)
    ax[i].plot(sorted_x_vs_tr[:, 0][49:], moving_average(sorted_x_vs_tr[:, 1]), c='k', label='Moving Average')
    ax[i].set_xscale('log', basex=2)
    ax[i].set_ylabel(r'$||T-T_{est}||_{F}$')
    ax[i].axhline(y=0, linestyle='--', linewidth=2, color='k', alpha=1)
    ax[i].set_title('Transition Matrix Estimation')
    ax[i].grid(linestyle='--')
    ax[i].tick_params(axis='both', which='major', grid_alpha=0.25)
