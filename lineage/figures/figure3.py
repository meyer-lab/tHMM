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
from ..Analyze import run_Analyze_over, run_Results_over
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
    Makes figures 3.
    """

    # Get list of axis objects
    ax, f = getSetup((21, 12))
#     f.subplot2grid(shape, loc, rowspan=1, colspan=1)
    x, bern_p0_est, bern_p1_est, bern_p0_true, bern_p1_true, gamma_a0_est, gamma_a1_est, gamma_a0_true, gamma_a1_true, gamma_scale0_est, gamma_scale1_est, gamma_scale0_true, gamma_scale1_true, accuracies, tr = accuracy_increased_cells()
    figure_maker(ax, x, bern_p0_est, bern_p1_est, bern_p0_true, bern_p1_true, gamma_a0_est, gamma_a1_est, gamma_a0_true, gamma_a1_true, gamma_scale0_est, gamma_scale1_est, gamma_scale0_true, gamma_scale1_true, accuracies, tr)
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
    bern_p0_true = 0.99
    gamma_a0_true = 20
    gamma_loc_true = 0
    gamma_scale0_true = 5

    # State 1 parameters "Susceptible"
    state1 = 1
    bern_p1_true = 0.88
    gamma_a1_true = 10
    gamma_scale1_true = 1

    state_obj0 = StateDistribution(state0, bern_p0_true, gamma_a0_true, gamma_loc_true, gamma_scale0_true)
    state_obj1 = StateDistribution(state1, bern_p1_true, gamma_a1_true, gamma_loc_true, gamma_scale1_true)
    E = [state_obj0, state_obj1]

    # Creating a list of populations to analyze over
    times = np.linspace(100, 1000, 12)
    list_of_populations = []
    for experiment_time in times:
        # Creating an unpruned and pruned lineage
        lineage = LineageTree(piiii, T, E, (2**12) - 1, experiment_time, prune_condition='both', prune_boolean=True)
        
        while len(lineage.output_lineage) < 16:
            del lineage
            lineage = LineageTree(piiii, T, E, (2**12) - 1, experiment_time, prune_condition='both', prune_boolean=True)

        # Adding populations into a holder for analysing
        list_of_populations.append([lineage])

    # Analyzing the lineages in the list of populations (parallelized function)
    output = run_Analyze_over(list_of_populations, 2)

    # Collecting the results of analyzing the lineages 
    results_holder = run_Results_over(output)
    
    # Collect necessary things to plot
    x = []
    bern_p0_est = []
    bern_p1_est = []
    gamma_a0_est = []
    gamma_a1_est = []
    gamma_scale0_est = []
    gamma_scale1_est = []
    accuracies = []
    tr = []
    
    for results_dict in results_holder:
        x.append(results_dict["total_number_of_cells"])
        accuracies.append(results_dict["accuracy_after_switching"])
        tr.append(results_dict["transition_matrix_norm"])
        bern_p0_est.append(results_dict["param_estimates"][0][0])
        bern_p1_est.append(results_dict["param_estimates"][1][0])
        gamma_a0_est.append(results_dict["param_estimates"][0][1])
        gamma_a1_est.append(results_dict["param_estimates"][1][1])
        gamma_scale0_est.append(results_dict["param_estimates"][0][3])
        gamma_scale1_est.append(results_dict["param_estimates"][1][3])

    return x, bern_p0_est, bern_p1_est, bern_p0_true, bern_p1_true, gamma_a0_est, gamma_a1_est, gamma_a0_true, gamma_a1_true, gamma_scale0_est, gamma_scale1_est, gamma_scale0_true, gamma_scale1_true, accuracies, tr

def figure_maker(ax, x, bern_p0_est, bern_p1_est, bern_p0_true, bern_p1_true, gamma_a0_est, gamma_a1_est, gamma_a0_true, gamma_a1_true, gamma_scale0_est, gamma_scale1_est, gamma_scale0_true, gamma_scale1_true, accuracies, tr):
    """
    Makes figure 3.
    """
    i = 0
    ax[i].set_xlim((16, int(np.ceil(4 * max(x)))))
    ax[i].set_xlabel('Number of Cells')
    ax[i].scatter(x, bern_p0_est, c='#F9Cb9C', edgecolors='k', marker="o", alpha=0.5)
    ax[i].scatter(x, bern_p1_est, c='#A4C2F4', edgecolors='k', marker="o", alpha=0.5)
    ax[i].set_ylabel('Bernoulli $p$')
    ax[i].set_ylim([0.85, 1.01])
    ax[i].axhline(y=bern_p0_true, linestyle='--', linewidth=2, label='Resistant', color='#F9Cb9C', alpha=1)
    ax[i].axhline(y=bern_p1_true, linestyle='--', linewidth=2, label='Susceptible', color='#A4C2F4', alpha=1)
    ax[i].set_title(r'Bernoulli $p$')
    ax[i].grid(linestyle='--')
    ax[i].set_xscale('log', basex=2)
    ax[i].tick_params(axis='both', which='major', grid_alpha=0.25)

    i += 1
    ax[i].set_xlim((16, int(np.ceil(4 * max(x)))))
    ax[i].set_xlabel('Number of Cells')
    ax[i].scatter(x, gamma_a0_est, c='#F9Cb9C', edgecolors='k', marker="o", alpha=0.5)
    ax[i].scatter(x, gamma_a1_est, c='#A4C2F4', edgecolors='k', marker="o", alpha=0.5)
    ax[i].set_ylabel(r'Gamma $k$')
    ax[i].set_ylim([5, 25])
    ax[i].axhline(y=gamma_a0_true, linestyle='--', linewidth=2, label='Resistant', color='#F9Cb9C', alpha=1)
    ax[i].axhline(y=gamma_a1_true, linestyle='--', linewidth=2, label='Susceptible', color='#A4C2F4', alpha=1)
    ax[i].set_title(r'Gamma $k$')
    ax[i].grid(linestyle='--')
    ax[i].set_xscale('log', basex=2)
    ax[i].tick_params(axis='both', which='major', grid_alpha=0.25)

    i += 1
    ax[i].set_xlim((16, int(np.ceil(4 * max(x)))))
    ax[i].set_xlabel('Number of Cells')
    ax[i].scatter(x, gamma_scale0_est, c='#F9Cb9C', edgecolors='k', marker="o", alpha=0.5)
    ax[i].scatter(x, gamma_scale1_est, c='#A4C2F4', edgecolors='k', marker="o", alpha=0.5)
    ax[i].set_ylabel(r'Gamma $\theta$')
    ax[i].set_ylim([0, 7])
    ax[i].axhline(y=gamma_scale0_true, linestyle='--', linewidth=2, label='Resistant', color='#F9Cb9C', alpha=1)
    ax[i].axhline(y=gamma_scale1_true, linestyle='--', linewidth=2, label='Susceptible', color='#A4C2F4', alpha=1)
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
