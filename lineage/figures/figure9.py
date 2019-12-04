"""
File: figure9.py
Purpose: Generates figure 9.

Figure 9 is the accuracy and transition matrix parameter estimation for a group of pruned lineages with heterogeneity (two true states).
"""
import numpy as np
import matplotlib.pyplot as plt

from .figureCommon import getSetup
from ..Analyze import accuracy, Analyze
from ..LineageTree import LineageTree
from ..StateDistribution import StateDistribution


def makeFigure():
    """
    Makes figure 9.
    """

    # Get list of axis objects
    ax, f = getSetup((24, 12), (2, 3))
    x, accuracies, tr, pi = accuracy_increased_cells9()
    x2, bern_unpruned, bern_p0, bern_p1, gamma_a_unpruned, gamma_a0, gamma_a1, gamma_scale_unpruned, gamma_scale0, gamma_scale1 = accuracy_increased_cells10()
    figure_maker(ax, x, accuracies, tr, pi, x2, bern_unpruned, bern_p0, bern_p1, gamma_a_unpruned, gamma_a0, gamma_a1, gamma_scale_unpruned, gamma_scale0, gamma_scale1)
    f.tight_layout()

    return f


def accuracy_increased_cells10():
    """
    Calculates parameter estimation by increasing the number of cells in a lineage for a two-state model.
    """

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

    desired_num_cells = 2**9 - 1
    experiment_time = 50
    num_lineages = list(range(1, 10))
    list_of_lineages = []

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
        list_of_lineages.append(population)

    x = []
    bern_unpruned = []
    gamma_a_unpruned = []
    gamma_scale_unpruned = []

    for population in list_of_lineages:
        # Analyzing the lineages
        _, _, all_states, tHMMobj, _, _ = Analyze(population, 2)

        # Collecting the accuracies of the lineages
        acc1 = accuracy(tHMMobj, all_states)[0] * 100
        while acc1 < 50:
            # Analyzing the lineages
            _, _, all_states, tHMMobj, _, _ = Analyze(population, 2)
            # Collecting the accuracies of the lineages
            acc1 = accuracy(tHMMobj, all_states)[0] * 100

        # Collecting how many cells are in each lineage in each analysis
        num_cells_holder = [len(lineageObj.output_lineage) for lineageObj in population]
        x.append(sum(num_cells_holder))

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


def accuracy_increased_cells9():
    """
    Calculates accuracy and transition rate estimation over an increasing number of cells in a lineage for an pruned two-state model.
    """

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

    desired_num_cells = 2**9 - 1
    experiment_time = 50
    num_lineages = list(range(1, 10))
    list_of_lineages = []

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
        list_of_lineages.append(population)

    x = []
    accuracies = []
    tr = []
    pi = []

    for population in list_of_lineages:
        # Analyzing the lineages
        deltas, _, all_states, tHMMobj, _, _ = Analyze(population, 2)

        # Collecting the accuracies of the lineages
        acc1 = accuracy(tHMMobj, all_states)[0] * 100
        accuracies.append(acc1)

        # Collecting how many cells are in each lineage in each analysis
        num_cells_holder = [len(lineageObj.output_lineage) for lineageObj in population]
        x.append(sum(num_cells_holder))

        # Transition and pi estimates
        transition_mat = tHMMobj.estimate.T
        tr.append(np.linalg.norm(T - transition_mat))

        pi_mat = tHMMobj.estimate.pi
        pi.append(np.linalg.norm(piiii - pi_mat))

    return x, accuracies, tr, pi


def moving_average(a, n=15):
    """
    Calculates the moving average.
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def figure_maker(ax, x, accuracies, tr, pi, x2, bern_unpruned, bern_p0, bern_p1, gamma_a_unpruned, gamma_a0, gamma_a1, gamma_scale_unpruned, gamma_scale0, gamma_scale1):
    """
    Makes figure 8.
    """
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
    ax[i].plot(sorted_x_vs_acc[:, 0][14:], moving_average(sorted_x_vs_acc[:, 1]), c='k', label='Moving Average')
    ax[i].set_ylabel(r'Accuracy [\%]')
    ax[i].axhline(y=100, linestyle='--', linewidth=2, color='k', alpha=1)
    ax[i].set_title('State Assignment Accuracy')
    ax[i].grid(linestyle='--')
    ax[i].tick_params(axis='both', which='major', grid_alpha=0.25)

    i += 1
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x)))))
    ax[i].set_xlabel('Number of Cells')
    ax[i].scatter(x, tr, c='k', marker="o", edgecolors='k', alpha=0.25)
    ax[i].plot(sorted_x_vs_tr[:, 0][14:], moving_average(sorted_x_vs_tr[:, 1]), c='k', label='Moving Average')
    ax[i].set_ylabel(r'$||T-T_{est}||_{F}$')
    ax[i].axhline(y=0, linestyle='--', linewidth=2, color='k', alpha=1)
    ax[i].set_title('Transition Matrix Estimation')
    ax[i].grid(linestyle='--')
    ax[i].tick_params(axis='both', which='major', grid_alpha=0.25)

    i += 1
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x)))))
    ax[i].set_xlabel('Number of Cells')
    ax[i].scatter(x, pi, c='k', marker="o", edgecolors='k', alpha=0.25)
    ax[i].plot(sorted_x_vs_pi[:, 0][14:], moving_average(sorted_x_vs_pi[:, 1]), c='k', label='Moving Average')
    ax[i].set_ylabel(r'$||\pi-\pi_{est}||_{2}$')
    ax[i].axhline(y=0, linestyle='--', linewidth=2, color='k', alpha=1)
    ax[i].set_title(r'Initial Seeding Proportion Estimation')
    ax[i].grid(linestyle='--')
    ax[i].tick_params(axis='both', which='major', grid_alpha=0.25)

    res = [[i for i, j in bern_unpruned], [j for i, j in bern_unpruned]]
    ax[i].set_xlim((16, int(np.ceil(4 * max(x2)))))
    ax[i].set_xlabel('Number of Cells')
    ax[i].scatter(x2, res[0], c='#F9Cb9C', edgecolors='k', marker="o", alpha=0.5)
    ax[i].scatter(x2, res[1], c='#A4C2F4', edgecolors='k', marker="o", alpha=0.5)
    ax[i].set_ylabel('Bernoulli $p$')
    ax[i].set_ylim([0.85, 1.1])
    ax[i].axhline(y=bern_p0, linestyle='--', linewidth=2, label='Resistant', color='#F9Cb9C', alpha=1)
    ax[i].axhline(y=bern_p1, linestyle='--', linewidth=2, label='Susceptible', color='#A4C2F4', alpha=1)
    ax[i].set_title(r'Bernoulli $p$')
    ax[i].grid(linestyle='--')
    ax[i].set_xscale('log', basex=2)
    ax[i].tick_params(axis='both', which='major', grid_alpha=0.25)

    i += 1
    res = [[i for i, j in gamma_a_unpruned], [j for i, j in gamma_a_unpruned]]
    ax[i].set_xlim((16, int(np.ceil(4 * max(x2)))))
    ax[i].set_xlabel('Number of Cells')
    ax[i].scatter(x2, res[0], c='#F9Cb9C', edgecolors='k', marker="o", alpha=0.5)
    ax[i].scatter(x2, res[1], c='#A4C2F4', edgecolors='k', marker="o", alpha=0.5)
    ax[i].set_ylabel(r'Gamma $k$')
    ax[i].set_ylim([5, 25])
    ax[i].axhline(y=gamma_a0, linestyle='--', linewidth=2, label='Resistant', color='#F9Cb9C', alpha=1)
    ax[i].axhline(y=gamma_a1, linestyle='--', linewidth=2, label='Susceptible', color='#A4C2F4', alpha=1)
    ax[i].set_title(r'Gamma $k$')
    ax[i].grid(linestyle='--')
    ax[i].set_xscale('log', basex=2)
    ax[i].tick_params(axis='both', which='major', grid_alpha=0.25)

    i += 1
    res = [[i for i, j in gamma_scale_unpruned], [j for i, j in gamma_scale_unpruned]]
    ax[i].set_xlim((16, int(np.ceil(4 * max(x2)))))
    ax[i].set_xlabel('Number of Cells')
    ax[i].scatter(x2, res[0], c='#F9Cb9C', edgecolors='k', marker="o", alpha=0.5)
    ax[i].scatter(x2, res[1], c='#A4C2F4', edgecolors='k', marker="o", alpha=0.5)
    ax[i].set_ylabel(r'Gamma $\theta$')
    ax[i].set_ylim([0, 7])
    ax[i].axhline(y=gamma_scale0, linestyle='--', linewidth=2, label='Resistant', color='#F9Cb9C', alpha=1)
    ax[i].axhline(y=gamma_scale1, linestyle='--', linewidth=2, label='Susceptible', color='#A4C2F4', alpha=1)
    ax[i].set_title(r'Gamma $\theta$')
    ax[i].grid(linestyle='--')
    ax[i].set_xscale('log', basex=2)
    ax[i].tick_params(axis='both', which='major', grid_alpha=0.25)
    ax[i].legend()
