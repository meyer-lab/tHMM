"""
This creates Figure 6.
"""
import copy as cp
import numpy as np
from .figureCommon import getSetup
from ..Analyze import accuracy, accuracyG, Analyze
from ..LineageTree import LineageTree
from ..StateDistribution import StateDistribution
from ..StateDistribution2 import StateDistribution2


def makeFigure():
    """ Main figure generating function for Fig. 6 """
    ax, f = getSetup((30, 10), (2, 6))

    x_unpruned, accuracies_unpruned, bern_unpruned, bern_p0, bern_p1, gamma_a_unpruned, gamma_a0, gamma_a1, gamma_scale_unpruned, gamma_scale0, gamma_scale1, x_pruned, accuracies_pruned, bern_pruned, gamma_a_pruned, gamma_scale_pruned, tr_unprunedNorm, tr_prunedNorm, pi_unprunedNorm, pi_prunedNorm = accuracy_increased_cells()
    figure_maker(
        ax,
        x_unpruned,
        accuracies_unpruned,
        bern_unpruned,
        bern_p0,
        bern_p1,
        gamma_a_unpruned,
        gamma_a0,
        gamma_a1,
        gamma_scale_unpruned,
        gamma_scale0,
        gamma_scale1,
        x_pruned,
        accuracies_pruned,
        bern_pruned,
        gamma_a_pruned,
        gamma_scale_pruned,
        tr_unprunedNorm,
        tr_prunedNorm,
        pi_unprunedNorm,
        pi_prunedNorm)

    return f



# Main figure generating function for Fig. 6 if we have G1 and G2 phase
# x_unpruned, accuracies_unpruned, bern_unpruned, bern_p0, bern_p1, gamma_aG1_unpruned, gamma_aG2_unpruned, gamma_aG11, gamma_aG12, gamma_aG21, gamma_aG22, gamma_scaleG1_unpruned, gamma_scaleG2_unpruned, gamma_scaleG11, gamma_scaleG12, gamma_scaleG21, gamma_scaleG22, x_pruned, accuracies_pruned, bern_pruned, gamma_aG1_pruned, gamma_scaleG1_pruned, gamma_aG2_pruned, gamma_scaleG2_pruned, tr_unprunedNorm, tr_prunedNorm, pi_unprunedNorm, pi_prunedNorm = accuracy_increased_cellsG()
# figure_makerG(
#     ax, x_unpruned, accuracies_unpruned, bern_unpruned, bern_p0, bern_p1, gamma_aG1_unpruned, gamma_aG2_unpruned, gamma_aG11, gamma_aG12, gamma_aG21, gamma_aG22, gamma_scaleG1_unpruned, gamma_scaleG2_unpruned, gamma_scaleG11, gamma_scaleG12, gamma_scaleG21, gamma_scaleG22, x_pruned, accuracies_pruned, bern_pruned, gamma_aG1_pruned, gamma_scaleG1_pruned, gamma_aG2_pruned, gamma_scaleG2_pruned, tr_unprunedNorm, tr_prunedNorm, pi_unprunedNorm, pi_prunedNorm)


# -------------------- Figure 6


def accuracy_increased_cells():
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

    desired_num_cells = np.logspace(5, 10, num=10, base=2.0)
    desired_num_cells = [num_cell - 1 for num_cell in desired_num_cells]

    x_unpruned = []
    x_pruned = []
    accuracies_unpruned = []
    accuracies_pruned = []
    bern_unpruned = []
    gamma_a_unpruned = []
    gamma_loc_unpruned = []
    gamma_scale_unpruned = []
    bern_pruned = []
    gamma_a_pruned = []
    gamma_loc_pruned = []
    gamma_scale_pruned = []
    tr_unprunedNorm = []
    tr_prunedNorm = []
    pi_unprunedNorm = []
    pi_prunedNorm = []

    for num in desired_num_cells:
        # Creating an unpruned and pruned lineage
        lineage_unpruned = LineageTree(pi, T, E, num, prune_boolean=False)
        # if the length of the pruned lineage tree is less than 5 cells, don't analyze either the pruned
        # or the unpruned lineage and skip
        if lineage_unpruned.__len__(True) <= 5:
            continue
        lineage_pruned = cp.deepcopy(lineage_unpruned)
        lineage_pruned.prune_boolean = True

        # Setting then into a list or a population of lineages and collecting the length of each lineage
        X1 = [lineage_unpruned]
        x_unpruned.append(len(lineage_unpruned.output_lineage))
        X2 = [lineage_pruned]
        x_pruned.append(len(lineage_pruned.output_lineage))

        # Analyzing the lineages
        deltas, _, all_states, tHMMobj, _, _ = Analyze(X1, 2)
        deltas2, _, all_states2, tHMMobj2, _, _ = Analyze(X2, 2)

        # Collecting the accuracies of the lineages
        acc1 = accuracy(tHMMobj, all_states)[0]
        acc2 = accuracy(tHMMobj2, all_states2)[0]
        accuracies_unpruned.append(acc1)
        accuracies_pruned.append(acc2)

        # Collecting the parameter estimations
        bern_p_total = ()
        gamma_a_total = ()
        gamma_loc_total = ()
        gamma_scale_total = ()
        bern_p_total2 = ()
        gamma_a_total2 = ()
        gamma_loc_total2 = ()
        gamma_scale_total2 = ()
        for state in range(tHMMobj.numStates):
            bern_p_total += (tHMMobj.estimate.E[state].bern_p,)
            gamma_a_total += (tHMMobj.estimate.E[state].gamma_a,)
            gamma_loc_total += (tHMMobj.estimate.E[state].gamma_loc,)
            gamma_scale_total += (tHMMobj.estimate.E[state].gamma_scale,)

            bern_p_total2 += (tHMMobj2.estimate.E[state].bern_p,)
            gamma_a_total2 += (tHMMobj2.estimate.E[state].gamma_a,)
            gamma_loc_total2 += (tHMMobj2.estimate.E[state].gamma_loc,)
            gamma_scale_total2 += (tHMMobj2.estimate.E[state].gamma_scale,)

        bern_unpruned.append(bern_p_total)
        gamma_a_unpruned.append(gamma_a_total)
        gamma_loc_unpruned.append(gamma_loc_total)
        gamma_scale_unpruned.append(gamma_scale_total)
        bern_pruned.append(bern_p_total2)
        gamma_a_pruned.append(gamma_a_total2)
        gamma_loc_pruned.append(gamma_loc_total2)
        gamma_scale_pruned.append(gamma_scale_total2)

    # Transition and Pi estimates
        transition_mat_unpruned = tHMMobj.estimate.T  # unpruned
        transition_mat_pruned = tHMMobj2.estimate.T  # pruned

        temp1 = T - transition_mat_unpruned
        temp2 = T - transition_mat_pruned
        tr_unprunedNorm.append(np.linalg.norm(temp1))
        tr_prunedNorm.append(np.linalg.norm(temp2))

        pi_mat_unpruned = tHMMobj.estimate.pi
        pi_mat_pruned = tHMMobj2.estimate.pi
        t1 = pi - pi_mat_unpruned
        t2 = pi - pi_mat_pruned
        pi_unprunedNorm.append(np.linalg.norm(t1))
        pi_prunedNorm.append(np.linalg.norm(t2))

    return x_unpruned, accuracies_unpruned, bern_unpruned, bern_p0, bern_p1, gamma_a_unpruned, gamma_a0, gamma_a1, gamma_scale_unpruned, gamma_scale0, gamma_scale1, x_pruned, accuracies_pruned, bern_pruned, gamma_a_pruned, gamma_scale_pruned, tr_unprunedNorm, tr_prunedNorm, pi_unprunedNorm, pi_prunedNorm


def figure_maker(ax, x_unpruned, accuracies_unpruned, bern_unpruned, bern_p0, bern_p1, gamma_a_unpruned, gamma_a0, gamma_a1, gamma_scale_unpruned, gamma_scale0,
                 gamma_scale1, x_pruned, accuracies_pruned, bern_pruned, gamma_a_pruned, gamma_scale_pruned, tr_unprunedNorm, tr_prunedNorm, pi_unprunedNorm, pi_prunedNorm):

    font = 11
    font2 = 10
    i = 0
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x_unpruned)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)
    ax[i].set_ylim(0, 110)
    ax[i].scatter(x_unpruned, accuracies_unpruned, c='k', marker="o", label='Accuracy', alpha=0.3)
    ax[i].axhline(y=100, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='k', alpha=0.6)  # linestyle is dashdotdotted
    ax[i].set_ylabel(r'Accuracy (\%)', rotation=90, fontsize=font2)
    ax[i].get_yticks()
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].set_title('State Assignment Accuracy', fontsize=font)

    i += 1
    res = [[i for i, j in bern_unpruned], [j for i, j in bern_unpruned]]
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x_unpruned)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)
    ax[i].scatter(x_unpruned, res[0], c='b', marker="o", label='Susceptible Unpruned', alpha=0.5)
    ax[i].scatter(x_unpruned, res[1], c='r', marker="o", label='Resistant Unpruned', alpha=0.5)
    ax[i].set_ylabel('Bern $p$', rotation=90, fontsize=font2)
    ax[i].axhline(y=bern_p0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b', alpha=0.6)
    ax[i].axhline(y=bern_p1, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[i].set_title('Bernoulli', fontsize=font)
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].legend(loc='best', framealpha=0.3)

    i += 1
    res = [[i for i, j in gamma_a_unpruned], [j for i, j in gamma_a_unpruned]]
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x_unpruned)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)
    ax[i].scatter(x_unpruned, res[0], c='b', marker="o", label='Susceptible Unpruned', alpha=0.5)
    ax[i].scatter(x_unpruned, res[1], c='r', marker="o", label='Resistant Unpruned', alpha=0.5)
    ax[i].set_ylabel(r'Gamma a $\beta$', rotation=90, fontsize=font2)
    ax[i].axhline(y=gamma_a0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b', alpha=0.6)
    ax[i].axhline(y=gamma_a1, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[i].set_title('Gamma', fontsize=font)
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].legend(loc='best', framealpha=0.3)

    i += 1
    res = [[i for i, j in gamma_scale_unpruned], [j for i, j in gamma_scale_unpruned]]
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x_unpruned)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)
    ax[i].scatter(x_unpruned, res[0], c='b', marker="o", label='Susceptible Unpruned', alpha=0.5)
    ax[i].scatter(x_unpruned, res[1], c='r', marker="o", label='Resistant Unpruned', alpha=0.5)
    ax[i].set_ylabel(r'Gamma scale $\alpha$', rotation=90, fontsize=font2)
    ax[i].axhline(y=gamma_scale0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b', alpha=0.6)
    ax[i].axhline(y=gamma_scale1, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[i].set_title('Gamma', fontsize=font)
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].legend(loc='best', framealpha=0.3)

    i += 1
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x_unpruned)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)

    ax[i].scatter(x_unpruned, tr_unprunedNorm, c='k', marker="o", label=' Unpruned', alpha=0.5)
    ax[i].set_ylabel(r'$||T-T_{est}||_{F}$', rotation=90, fontsize=font2)
    ax[i].axhline(y=0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='k', alpha=0.6)
    ax[i].set_title('Norm Transition', fontsize=font)
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].legend(loc='best', framealpha=0.3)

    i += 1
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x_unpruned)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)
    ax[i].scatter(x_unpruned, pi_unprunedNorm, c='k', marker="o", label=' Unpruned', alpha=0.5)
    ax[i].set_ylabel(r'$||\pi-\pi_{est}||_{2}$', rotation=90, fontsize=font2)
    ax[i].axhline(y=0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='k', alpha=0.6)
    ax[i].set_title('Norm Pi', fontsize=font)
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].legend(loc='best', framealpha=0.3)

    i += 1
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x_unpruned)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)

    ax[i].set_ylim(0, 110)
    ax[i].scatter(x_pruned, accuracies_pruned, c='k', marker="o", label='Accuracy', alpha=0.3)
    ax[i].axhline(y=100, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='k', alpha=0.6)  # linestyle is dashdotdotted
    ax[i].set_ylabel(r'Accuracy (\%)', rotation=90, fontsize=font2)
    ax[i].get_yticks()
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].set_title('State Assignment Accuracy', fontsize=font)

    i += 1
    res = [[i for i, j in bern_pruned], [j for i, j in bern_pruned]]
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x_unpruned)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)
    ax[i].scatter(x_pruned, res[0], c='b', marker="o", label='Susceptible Pruned', alpha=0.5)
    ax[i].scatter(x_pruned, res[1], c='r', marker="o", label='Resistant Pruned', alpha=0.5)
    ax[i].set_ylabel('Bern $p$', rotation=90, fontsize=font2)
    ax[i].axhline(y=bern_p0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b', alpha=0.6)
    ax[i].axhline(y=bern_p1, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[i].set_title('Bernoulli', fontsize=font)
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].legend(loc='best', framealpha=0.3)

    i += 1
    res = [[i for i, j in gamma_a_pruned], [j for i, j in gamma_a_pruned]]
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x_unpruned)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)
    ax[i].scatter(x_pruned, res[0], c='b', marker="o", label='Susceptible Pruned', alpha=0.5)
    ax[i].scatter(x_pruned, res[1], c='r', marker="o", label='Resistant Pruned', alpha=0.5)
    ax[i].set_ylabel(r'Gamma a $\beta$', rotation=90, fontsize=font2)
    ax[i].axhline(y=gamma_a0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b', alpha=0.6)
    ax[i].axhline(y=gamma_a1, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[i].set_title('Gamma', fontsize=font)
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].legend(loc='best', framealpha=0.3)

    i += 1
    res = [[i for i, j in gamma_scale_pruned], [j for i, j in gamma_scale_pruned]]
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x_unpruned)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)
    ax[i].scatter(x_pruned, res[0], c='b', marker="o", label='Susceptible Pruned', alpha=0.5)
    ax[i].scatter(x_pruned, res[1], c='r', marker="o", label='Resistant Pruned', alpha=0.5)
    ax[i].set_ylabel(r'Gamma scale $\alpha$', rotation=90, fontsize=font2)
    ax[i].axhline(y=gamma_scale0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b', alpha=0.6)
    ax[i].axhline(y=gamma_scale1, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[i].set_title('Gamma', fontsize=font)
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].legend(loc='best', framealpha=0.3)

    i += 1
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x_pruned)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)
    ax[i].scatter(x_pruned, tr_prunedNorm, c='k', marker="o", label=' Pruned', alpha=0.5)
    ax[i].set_ylabel(r'$||T-T_{est}||_{F}$', rotation=90, fontsize=font2)
    ax[i].axhline(y=0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='k', alpha=0.6)
    ax[i].set_title('Norm Transition', fontsize=font)
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].legend(loc='best', framealpha=0.3)

    i += 1
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x_pruned)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)
    ax[i].scatter(x_pruned, pi_prunedNorm, c='k', marker="o", label=' Pruned', alpha=0.5)
    ax[i].set_ylabel(r'$||\pi-\pi_{est}||_{2}$', rotation=90, fontsize=font2)
    ax[i].axhline(y=0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='k', alpha=0.6)
    ax[i].set_title('Norm Pi', fontsize=font)
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].legend(loc='best', framealpha=0.3)


# ----------- Figure 6 for G1G2

def accuracy_increased_cellsG():
    """ Calclates accuracy and parameter estimation by increasing the number of cells in a lineage for a two-state model. """

    # pi: the initial probability vector
    pi = np.array([0.6, 0.4], dtype="float")

    # T: transition probability matrix
    T = np.array([[0.85, 0.15],
                  [0.15, 0.85]], dtype="float")

    # State 0 parameters "Resistant"
    state0 = 0
    bern_p0 = 0.99
    gamma_aG11 = 10
    gamma_loc = 0
    gamma_scaleG11 = 2.0
    gamma_aG21 = 15
    gamma_scaleG21 = 2.0

    # State 1 parameters "Susceptible"
    state1 = 1
    bern_p1 = 0.88
    gamma_aG12 = 7
    gamma_scaleG12 = 1.0
    gamma_aG22 = 18
    gamma_scaleG22 = 1.0

    state_obj0 = StateDistribution2(state0, bern_p0, gamma_aG11, gamma_loc, gamma_scaleG11, gamma_aG21, gamma_scaleG21)
    state_obj1 = StateDistribution2(state1, bern_p1, gamma_aG12, gamma_loc, gamma_scaleG12, gamma_aG22, gamma_scaleG22)

    E = [state_obj0, state_obj1]
    # the key part in this function
    desired_num_cells = np.logspace(7, 10, num=10, base=2.0)
    desired_num_cells = [num_cell - 1 for num_cell in desired_num_cells]

    x_unpruned = []
    x_pruned = []
    accuracies_unpruned = []
    accuracies_pruned = []
    bern_unpruned = []
    gamma_aG1_unpruned = []
    gamma_loc_unpruned = []
    gamma_scaleG1_unpruned = []
    gamma_aG2_unpruned = []
    gamma_scaleG2_unpruned = []
    bern_pruned = []
    gamma_aG1_pruned = []
    gamma_scaleG1_pruned = []
    gamma_aG2_pruned = []
    gamma_loc_pruned = []
    gamma_scaleG2_pruned = []
    tr_unprunedNorm = []
    tr_prunedNorm = []
    pi_unprunedNorm = []
    pi_prunedNorm = []

    for num in desired_num_cells:
        lineage_unpruned = LineageTree(pi, T, E, num, prune_boolean=False)
        lineage_pruned = cp.deepcopy(lineage_unpruned)
        lineage_pruned.prune_boolean = True
        # Setting then into a list or a population of lineages and collecting the length of each lineage
        X1 = [lineage_unpruned]
        x_unpruned.append(len(lineage_unpruned.output_lineage))
        X2 = [lineage_pruned]
        x_pruned.append(len(lineage_pruned.output_lineage))

        print("unpruned")
        deltas, _, all_states, tHMMobj, _, _ = Analyze(X1, 2)
        print("pruned")
        deltas2, _, all_states2, tHMMobj2, _, _ = Analyze(X2, 2)
        acc1 = accuracyG(tHMMobj, all_states)[0]  # for pruned
        acc2 = accuracyG(tHMMobj2, all_states2)[0]  # for unpruned
        accuracies_unpruned.append(acc1)
        accuracies_pruned.append(acc2)

        bern_p_total = ()
        gamma_aG1_total = ()
        gamma_loc_total = ()
        gamma_scaleG1_total = ()
        gamma_aG2_total = ()
        gamma_scaleG2_total = ()
        bern_p_total2 = ()
        gamma_aG1_total2 = ()
        gamma_loc_total2 = ()
        gamma_scaleG1_total2 = ()
        gamma_aG2_total2 = ()
        gamma_scaleG2_total2 = ()
        for state in range(tHMMobj.numStates):
            # upruned
            bern_p_total += (tHMMobj.estimate.E[state].bern_p,)
            gamma_aG1_total += (tHMMobj.estimate.E[state].gamma_a1,)
            gamma_loc_total += (tHMMobj.estimate.E[state].gamma_loc,)
            gamma_scaleG1_total += (tHMMobj.estimate.E[state].gamma_scale1,)
            gamma_aG2_total += (tHMMobj.estimate.E[state].gamma_a2,)
            gamma_scaleG2_total += (tHMMobj.estimate.E[state].gamma_scale2,)

            # pruned
            bern_p_total2 += (tHMMobj2.estimate.E[state].bern_p,)
            gamma_aG1_total2 += (tHMMobj2.estimate.E[state].gamma_a1,)
            gamma_loc_total2 += (tHMMobj2.estimate.E[state].gamma_loc,)
            gamma_scaleG1_total2 += (tHMMobj2.estimate.E[state].gamma_scale1,)
            gamma_aG2_total2 += (tHMMobj2.estimate.E[state].gamma_a2,)
            gamma_scaleG2_total2 += (tHMMobj2.estimate.E[state].gamma_scale2,)

        # unpruned
        bern_unpruned.append(bern_p_total)
        gamma_aG1_unpruned.append(gamma_aG1_total)
        gamma_loc_unpruned.append(gamma_loc_total)
        gamma_scaleG1_unpruned.append(gamma_scaleG1_total)
        gamma_aG2_unpruned.append(gamma_aG2_total)
        gamma_scaleG2_unpruned.append(gamma_scaleG2_total)

        # pruned
        bern_pruned.append(bern_p_total2)
        gamma_aG1_pruned.append(gamma_aG1_total2)
        gamma_loc_pruned.append(gamma_loc_total2)
        gamma_scaleG1_pruned.append(gamma_scaleG1_total2)
        gamma_aG2_pruned.append(gamma_aG2_total2)
        gamma_scaleG2_pruned.append(gamma_scaleG2_total2)

        # Transition and Pi estimates
        transition_mat_unpruned = tHMMobj.estimate.T  # unpruned
        transition_mat_pruned = tHMMobj2.estimate.T  # pruned

        temp1 = T - transition_mat_unpruned
        temp2 = T - transition_mat_pruned
        tr_unprunedNorm.append(np.linalg.norm(temp1))
        tr_prunedNorm.append(np.linalg.norm(temp2))

        pi_mat_unpruned = tHMMobj.estimate.pi
        pi_mat_pruned = tHMMobj2.estimate.pi
        t1 = pi - pi_mat_unpruned
        t2 = pi - pi_mat_pruned
        pi_unprunedNorm.append(np.linalg.norm(t1))
        pi_prunedNorm.append(np.linalg.norm(t2))

    return x_unpruned, accuracies_unpruned, bern_unpruned, bern_p0, bern_p1, gamma_aG1_unpruned, gamma_aG2_unpruned, gamma_aG11, gamma_aG12, gamma_aG21, gamma_aG22, gamma_scaleG1_unpruned, gamma_scaleG2_unpruned, gamma_scaleG11, gamma_scaleG12, gamma_scaleG21, gamma_scaleG22, x_pruned, accuracies_pruned, bern_pruned, gamma_aG1_pruned, gamma_scaleG1_pruned, gamma_aG2_pruned, gamma_scaleG2_pruned, tr_unprunedNorm, tr_prunedNorm, pi_unprunedNorm, pi_prunedNorm


# ------------- figure for G1G2

def figure_makerG(ax, x_unpruned, accuracies_unpruned, bern_unpruned, bern_p0, bern_p1, gamma_aG1_unpruned, gamma_aG2_unpruned, gamma_aG11, gamma_aG12, gamma_aG21, gamma_aG22, gamma_scaleG1_unpruned, gamma_scaleG2_unpruned, gamma_scaleG11,
                  gamma_scaleG12, gamma_scaleG21, gamma_scaleG22, x_pruned, accuracies_pruned, bern_pruned, gamma_aG1_pruned, gamma_scaleG1_pruned, gamma_aG2_pruned, gamma_scaleG2_pruned, tr_unprunedNorm, tr_prunedNorm, pi_unprunedNorm, pi_prunedNorm):

    font = 11
    font2 = 10
    i = 0
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x_unpruned)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)
    ax[i].set_ylim(0, 110)
    ax[i].scatter(x_unpruned, accuracies_unpruned, c='k', marker="o", label='Accuracy', alpha=0.3)
    ax[i].axhline(y=100, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='k', alpha=0.6)  # linestyle is dashdotdotted
    ax[i].set_ylabel(r'Accuracy (\%)', rotation=90, fontsize=font2)
    ax[i].get_yticks()
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].set_title('State Assignment Accuracy', fontsize=font)

    i += 1
    res = [[i for i, j in bern_unpruned], [j for i, j in bern_unpruned]]
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x_unpruned)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)
    ax[i].scatter(x_unpruned, res[0], c='b', marker="o", label='Susceptible Unpruned', alpha=0.5)
    ax[i].scatter(x_unpruned, res[1], c='r', marker="o", label='Resistant Unpruned', alpha=0.5)
    ax[i].set_ylabel('Bern $p$', rotation=90, fontsize=font2)
    ax[i].axhline(y=bern_p0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b', alpha=0.6)
    ax[i].axhline(y=bern_p1, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[i].set_title('Bernoulli', fontsize=font)
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].legend(loc='best', framealpha=0.3)

    i += 1
    res = [[i for i, j in gamma_aG1_unpruned], [j for i, j in gamma_aG1_unpruned]]
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x_unpruned)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)
    ax[i].scatter(x_unpruned, res[0], c='b', marker="o", label='Susceptible Unpruned', alpha=0.5)
    ax[i].scatter(x_unpruned, res[1], c='r', marker="o", label='Resistant Unpruned', alpha=0.5)
    ax[i].set_ylabel(r'Gamma a $\beta$', rotation=90, fontsize=font2)
    ax[i].axhline(y=gamma_aG11, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b', alpha=0.6)
    ax[i].axhline(y=gamma_aG12, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[i].set_title('Gamma G1', fontsize=font)
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].legend(loc='best', framealpha=0.3)

    i += 1
    res = [[i for i, j in gamma_scaleG1_unpruned], [j for i, j in gamma_scaleG1_unpruned]]
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x_unpruned)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)
    ax[i].scatter(x_unpruned, res[0], c='b', marker="o", label='Susceptible Unpruned', alpha=0.5)
    ax[i].scatter(x_unpruned, res[1], c='r', marker="o", label='Resistant Unpruned', alpha=0.5)
    ax[i].set_ylabel(r'Gamma scale $\alpha$', rotation=90, fontsize=font2)
    ax[i].axhline(y=gamma_scaleG11, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b', alpha=0.6)
    ax[i].axhline(y=gamma_scaleG12, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[i].set_title('Gamma G1', fontsize=font)
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].legend(loc='best', framealpha=0.3)

    i += 1
    res = [[i for i, j in gamma_aG2_unpruned], [j for i, j in gamma_aG2_unpruned]]
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x_unpruned)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)
    ax[i].scatter(x_unpruned, res[0], c='b', marker="o", label='Susceptible Unpruned', alpha=0.5)
    ax[i].scatter(x_unpruned, res[1], c='r', marker="o", label='Resistant Unpruned', alpha=0.5)
    ax[i].set_ylabel(r'Gamma a $\beta$', rotation=90, fontsize=font2)
    ax[i].axhline(y=gamma_aG21, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b', alpha=0.6)
    ax[i].axhline(y=gamma_aG22, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[i].set_title('Gamma G2', fontsize=font)
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].legend(loc='best', framealpha=0.3)

    i += 1
    res = [[i for i, j in gamma_scaleG2_unpruned], [j for i, j in gamma_scaleG2_unpruned]]
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x_unpruned)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)
    ax[i].scatter(x_unpruned, res[0], c='b', marker="o", label='Susceptible Unpruned', alpha=0.5)
    ax[i].scatter(x_unpruned, res[1], c='r', marker="o", label='Resistant Unpruned', alpha=0.5)
    ax[i].set_ylabel(r'Gamma scale $\alpha$', rotation=90, fontsize=font2)
    ax[i].axhline(y=gamma_scaleG21, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b', alpha=0.6)
    ax[i].axhline(y=gamma_scaleG22, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[i].set_title('Gamma G2', fontsize=font)
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].legend(loc='best', framealpha=0.3)

    i += 1
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x_unpruned)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)
    ax[i].scatter(x_unpruned, tr_unprunedNorm, c='k', marker="o", label=' Unpruned', alpha=0.5)
    ax[i].set_ylabel(r'$||T-T_{est}||_{F}$', rotation=90, fontsize=font2)
    ax[i].axhline(y=0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='k', alpha=0.6)
    ax[i].set_title('Norm Transition', fontsize=font)
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].legend(loc='best', framealpha=0.3)

    i += 1
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x_unpruned)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)
    ax[i].scatter(x_unpruned, pi_unprunedNorm, c='k', marker="o", label=' Unpruned', alpha=0.5)
    ax[i].set_ylabel(r'$||\pi-\pi_{est}||_{2}$', rotation=90, fontsize=font2)
    ax[i].axhline(y=0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='k', alpha=0.6)
    ax[i].set_title('Norm Pi', fontsize=font)
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].legend(loc='best', framealpha=0.3)

    i += 1
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x_pruned)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)
    ax[i].set_ylim(0, 110)
    ax[i].scatter(x_pruned, accuracies_pruned, c='k', marker="o", label='Accuracy', alpha=0.3)
    ax[i].axhline(y=100, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='k', alpha=0.6)  # linestyle is dashdotdotted
    ax[i].set_ylabel(r'Accuracy (\%)', rotation=90, fontsize=font2)
    ax[i].get_yticks()
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].set_title('State Assignment Accuracy', fontsize=font)

    i += 1
    res = [[i for i, j in bern_pruned], [j for i, j in bern_pruned]]
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x_pruned)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)
    ax[i].scatter(x_pruned, res[0], c='b', marker="o", label='Susceptible Pruned', alpha=0.5)
    ax[i].scatter(x_pruned, res[1], c='r', marker="o", label='Resistant Pruned', alpha=0.5)
    ax[i].set_ylabel('Bern $p$', rotation=90, fontsize=font2)
    ax[i].axhline(y=bern_p0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b', alpha=0.6)
    ax[i].axhline(y=bern_p1, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[i].set_title('Bernoulli', fontsize=font)
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].legend(loc='best', framealpha=0.3)

    i += 1
    res = [[i for i, j in gamma_aG1_pruned], [j for i, j in gamma_aG1_pruned]]
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x_pruned)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)
    ax[i].scatter(x_pruned, res[0], c='b', marker="o", label='Susceptible Pruned', alpha=0.5)
    ax[i].scatter(x_pruned, res[1], c='r', marker="o", label='Resistant Pruned', alpha=0.5)
    ax[i].set_ylabel(r'Gamma a $\beta$', rotation=90, fontsize=font2)
    ax[i].axhline(y=gamma_aG11, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b', alpha=0.6)
    ax[i].axhline(y=gamma_aG12, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[i].set_title('Gamma G1', fontsize=font)
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].legend(loc='best', framealpha=0.3)

    i += 1
    res = [[i for i, j in gamma_scaleG1_pruned], [j for i, j in gamma_scaleG1_pruned]]
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x_pruned)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)
    ax[i].scatter(x_pruned, res[0], c='b', marker="o", label='Susceptible Pruned', alpha=0.5)
    ax[i].scatter(x_pruned, res[1], c='r', marker="o", label='Resistant Pruned', alpha=0.5)
    ax[i].set_ylabel(r'Gamma scale $\alpha$', rotation=90, fontsize=font2)
    ax[i].axhline(y=gamma_scaleG11, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b', alpha=0.6)
    ax[i].axhline(y=gamma_scaleG12, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[i].set_title('Gamma G1', fontsize=font)
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].legend(loc='best', framealpha=0.3)

    i += 1
    res = [[i for i, j in gamma_aG2_pruned], [j for i, j in gamma_aG2_pruned]]
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x_pruned)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)
    ax[i].scatter(x_pruned, res[0], c='b', marker="o", label='Susceptible Pruned', alpha=0.5)
    ax[i].scatter(x_pruned, res[1], c='r', marker="o", label='Resistant Pruned', alpha=0.5)
    ax[i].set_ylabel(r'Gamma a $\beta$', rotation=90, fontsize=font2)
    ax[i].axhline(y=gamma_aG21, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b', alpha=0.6)
    ax[i].axhline(y=gamma_aG22, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[i].set_title('Gamma G2', fontsize=font)
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].legend(loc='best', framealpha=0.3)

    i += 1
    res = [[i for i, j in gamma_scaleG2_pruned], [j for i, j in gamma_scaleG2_pruned]]
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x_pruned)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)
    ax[i].scatter(x_pruned, res[0], c='b', marker="o", label='Susceptible Pruned', alpha=0.5)
    ax[i].scatter(x_pruned, res[1], c='r', marker="o", label='Resistant Pruned', alpha=0.5)
    ax[i].set_ylabel(r'Gamma scale $\alpha$', rotation=90, fontsize=font2)
    ax[i].axhline(y=gamma_scaleG21, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b', alpha=0.6)
    ax[i].axhline(y=gamma_scaleG22, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[i].set_title('Gamma G2', fontsize=font)
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].legend(loc='best', framealpha=0.3)
    i += 1
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x_pruned)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)
    ax[i].scatter(x_pruned, tr_prunedNorm, c='k', marker="o", label=' Pruned', alpha=0.5)
    ax[i].set_ylabel(r'$||T-T_{est}||_{F}$', rotation=90, fontsize=font2)
    ax[i].axhline(y=0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='k', alpha=0.6)
    ax[i].set_title('Norm Transition', fontsize=font)
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].legend(loc='best', framealpha=0.3)

    i += 1
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x_pruned)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)
    ax[i].scatter(x_pruned, pi_prunedNorm, c='k', marker="o", label=' Pruned', alpha=0.5)
    ax[i].set_ylabel(r'$||\pi-\pi_{est}||_{2}$', rotation=90, fontsize=font2)
    ax[i].axhline(y=0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='k', alpha=0.6)
    ax[i].set_title('Norm Pi', fontsize=font)
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].legend(loc='best', framealpha=0.3)
