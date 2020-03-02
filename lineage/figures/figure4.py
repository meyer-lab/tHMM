"""
File: figure4.py
Purpose: Generates figure 4.
Figure 4 analyzes heterogeneous (2 state), pruned (by both time and fate), populations of lineages
(more than one lineage per populations) with at least 10 cells per lineage over increasing
number of lineages per population.
"""
import numpy as np

from .figureCommon import getSetup, moving_average
from ..Analyze import run_Analyze_over, run_Results_over
from ..LineageTree import LineageTree
from ..StateDistribution import StateDistribution


def makeFigure():
    """
    Makes figure 9 and 10.
    """

    # Get list of axis objects
    ax, f = getSetup((24, 12), (2, 3))
    x, bern_p0_est, bern_p1_est, bern_p0_true, bern_p1_true, gamma_a0_est, gamma_a1_est, gamma_a0_true, gamma_a1_true, gamma_scale0_est, gamma_scale1_est, gamma_scale0_true, gamma_scale1_true, accuracies, tr, pi = accuracy_increased_cells()
    figure_maker(ax, x, bern_p0_est, bern_p1_est, bern_p0_true, bern_p1_true, gamma_a0_est, gamma_a1_est, gamma_a0_true,
                 gamma_a1_true, gamma_scale0_est, gamma_scale1_est, gamma_scale0_true, gamma_scale1_true, accuracies, tr, pi)

    return f


def accuracy_increased_cells():
    """
    Calculates parameter estimation by increasing the number of cells in a lineage for a two-state model.
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
    gamma_scale0_true = 5

    # State 1 parameters "Susceptible"
    state1 = 1
    bern_p1_true = 0.88
    gamma_a1_true = 10
    gamma_scale1_true = 1

    state_obj0 = StateDistribution(state0, bern_p0_true, gamma_a0_true, gamma_scale0_true)
    state_obj1 = StateDistribution(state1, bern_p1_true, gamma_a1_true, gamma_scale1_true)
    E = [state_obj0, state_obj1]

    desired_num_cells = 2**9 - 1
    experiment_time = 50

    # Creating a list of populations to analyze over
    num_lineages = list(range(1, 500))
    list_of_populations = []
    for num in num_lineages:
        population = []
        for _ in range(num):
            # Creating an unpruned and pruned lineage
            tmp_lineage = LineageTree(piiii, T, E, desired_num_cells, experiment_time, prune_condition='both', prune_boolean=True)
            if len(tmp_lineage.output_lineage) < 10:
                del tmp_lineage
                tmp_lineage = LineageTree(piiii, T, E, desired_num_cells, experiment_time, prune_condition='both', prune_boolean=True)
            population.append(tmp_lineage)
        # Adding populations into a holder for analysing
        list_of_populations.append(population)

    # Analyzing the lineages in the list of populations (parallelized function)
    output = run_Analyze_over(list_of_populations, 2, parallel=True)

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
    pi = []

    for results_dict in results_holder:
        x.append(results_dict["total_number_of_cells"])
        accuracies.append(results_dict["accuracy_after_switching"])
        tr.append(results_dict["transition_matrix_norm"])
        pi.append(results_dict["pi_vector_norm"])
        bern_p0_est.append(results_dict["param_estimates"][0][0])
        bern_p1_est.append(results_dict["param_estimates"][1][0])
        gamma_a0_est.append(results_dict["param_estimates"][0][1])
        gamma_a1_est.append(results_dict["param_estimates"][1][1])
        gamma_scale0_est.append(results_dict["param_estimates"][0][2])
        gamma_scale1_est.append(results_dict["param_estimates"][1][2])

    return x, bern_p0_est, bern_p1_est, bern_p0_true, bern_p1_true, gamma_a0_est, gamma_a1_est, gamma_a0_true, gamma_a1_true, gamma_scale0_est, gamma_scale1_est, gamma_scale0_true, gamma_scale1_true, accuracies, tr, pi


def figure_maker(ax, x, bern_p0_est, bern_p1_est, bern_p0_true, bern_p1_true, gamma_a0_est, gamma_a1_est, gamma_a0_true,
                 gamma_a1_true, gamma_scale0_est, gamma_scale1_est, gamma_scale0_true, gamma_scale1_true, accuracies, tr, pi):
    """
    Makes figure 4.
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

    x_vs_pi = np.column_stack((x, pi))
    sorted_x_vs_pi = x_vs_pi[np.argsort(x_vs_pi[:, 0])]

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

    i += 1
    ax[i].set_xlim((16, int(np.ceil(4 * max(x)))))
    ax[i].set_xlabel('Number of Cells')
    ax[i].scatter(x, pi, c='k', marker="o", edgecolors='k', alpha=0.25)
    ax[i].plot(sorted_x_vs_pi[:, 0][49:], moving_average(sorted_x_vs_pi[:, 1]), c='k', label='Moving Average')
    ax[i].set_xscale('log', basex=2)
    ax[i].set_ylabel(r'$||T-T_{est}||_{F}$')
    ax[i].axhline(y=0, linestyle='--', linewidth=2, color='k', alpha=1)
    ax[i].set_title('Transition Matrix Estimation')
    ax[i].grid(linestyle='--')
    ax[i].tick_params(axis='both', which='major', grid_alpha=0.25)
