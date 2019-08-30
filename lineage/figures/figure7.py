"""
This creates Figure 7. AIC Figure.
"""
from .figureCommon import getSetup
from ..Analyze import accuracy, Analyze, accuracy_for_lineages
from ..LineageTree import LineageTree
from ..StateDistribution import StateDistribution

import numpy as np
import copy as cp
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


def makeFigure():
    """ Main figure generating function for Fig. 6 """
    ax, f = getSetup((20, 10), (2, 4))

    #num_lineages, unprunedNewAcc, bern_unpruned, bern_p0, bern_p1, gamma_a_unpruned, gamma_a0, gamma_a1, gamma_scale_unpruned, gamma_scale0, gamma_scale1, prunedNewAcc, bern_pruned, gamma_a_pruned, gamma_scale_pruned = accuracy_increased_lineages()
    #figure_maker(ax, num_lineages, unprunedNewAcc, bern_unpruned, bern_p0, bern_p1, gamma_a_unpruned, gamma_a0, gamma_a1, gamma_scale_unpruned, gamma_scale0, gamma_scale1, prunedNewAcc, bern_pruned, gamma_a_pruned, gamma_scale_pruned)

    f.tight_layout()
    return f


# -------------------- Figure 7
def accuracy_increased_lineages():
    """ Calclates accuracy and parameter estimation by increasing the number of lineages. """

    # pi: the initial probability vector
    pi = np.array([0.5, 0.5], dtype="float")

    # T: transition probability matrix
    T = np.array([[0.99, 0.01],
                  [0.15, 0.85]], dtype='float')

    # State 0 parameters "Resistant"
    state0 = 0
    bern_p0 = 0.99
    gamma_a0 = 20
    gamma_scale0 = 5

    # State 1 parameters "Susciptible"
    state1 = 1
    bern_p1 = 0.8
    gamma_a1 = 10
    gamma_scale1 = 1

    state_obj0 = StateDistribution(state0, bern_p0, gamma_a0, gamma_scale0)
    state_obj1 = StateDistribution(state1, bern_p1, gamma_a1, gamma_scale1)

    E = [state_obj0, state_obj1]

    desired_num_cells = 2**6 - 1
    # increasing number of lineages from 1 to 10 and calculating accuracy and estimate parameters for both pruned and unpruned lineages.
    num_lineages = list(range(1, 5))

    accuracies_unpruned = []
    accuracies_pruned = []
    bern_unpruned = []
    gamma_a_unpruned = []
    gamma_scale_unpruned = []
    bern_pruned = []
    gamma_a_pruned = []
    gamma_scale_pruned = []

    prunedNewAcc = []
    unprunedNewAcc = []
    X_p = []
    X_unp = []
    for num in num_lineages:
        lineage_unpruned = LineageTree(pi, T, E, desired_num_cells, prune_boolean=False)
        lineage_pruned = cp.deepcopy(lineage_unpruned)
        lineage_pruned.prune_boolean = True

        X_unp.append(lineage_unpruned)
        X_p.append(lineage_pruned)
        deltas, state_ptrs, all_states, tHMMobj, NF, LL = Analyze(X_unp, 2)
        deltas2, state_ptrs2, all_states2, tHMMobj2, NF2, LL2 = Analyze(X_p, 2)
        acc1 = accuracy_for_lineages(tHMMobj, all_states)
        acc2 = accuracy_for_lineages(tHMMobj2, all_states2)
        accuracies_unpruned.append(acc1)
        accuracies_pruned.append(acc2)

        bern_p_total = ()
        gamma_a_total = ()
        gamma_scale_total = ()
        bern_p_total2 = ()
        gamma_a_total2 = ()
        gamma_scale_total2 = ()

        for state in range(tHMMobj.numStates):
            bern_p_total += (tHMMobj.estimate.E[state].bern_p,)
            gamma_a_total += (tHMMobj.estimate.E[state].gamma_a,)
            gamma_scale_total += (tHMMobj.estimate.E[state].gamma_scale,)

            bern_p_total2 += (tHMMobj2.estimate.E[state].bern_p,)
            gamma_a_total2 += (tHMMobj2.estimate.E[state].gamma_a,)
            gamma_scale_total2 += (tHMMobj2.estimate.E[state].gamma_scale,)

        bern_unpruned.append(bern_p_total)
        gamma_a_unpruned.append(gamma_a_total)
        gamma_scale_unpruned.append(gamma_scale_total)
        bern_pruned.append(bern_p_total2)
        gamma_a_pruned.append(gamma_a_total2)
        gamma_scale_pruned.append(gamma_scale_total2)

    for i in range(len(accuracies_unpruned)):
        unprunedNewAcc.append(sum(accuracies_unpruned[i]) / (i + 1))
        prunedNewAcc.append(sum(accuracies_pruned[i]) / (i + 1))

    return num_lineages, unprunedNewAcc, bern_unpruned, bern_p0, bern_p1, gamma_a_unpruned, gamma_a0, gamma_a1, gamma_scale_unpruned, gamma_scale0, gamma_scale1, prunedNewAcc, bern_pruned, gamma_a_pruned, gamma_scale_pruned


def figure_maker(ax, num_lineages, unprunedNewAcc, bern_unpruned, bern_p0, bern_p1, gamma_a_unpruned, gamma_a0, gamma_a1,
                 gamma_scale_unpruned, gamma_scale0, gamma_scale1, prunedNewAcc, bern_pruned, gamma_a_pruned, gamma_scale_pruned):
    x = num_lineages
    font = 11
    font2 = 10
    ax[0].set_xlim((0, int(np.ceil(1.1 * max(x)))))
    ax[0].set_xlabel('Number of Lineages', fontsize=font2)
    ax[0].set_ylim(0, 110)
    ax[0].scatter(x, unprunedNewAcc, c='k', marker="o", label='Accuracy', alpha=0.3)
    ax[0].axhline(y=100, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='k', alpha=0.6)  # linestyle is dashdotdotted
    ax[0].set_ylabel(r'Accuracy (\%)', rotation=90, fontsize=font2)
    ax[0].get_yticks()
    ax[0].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[0].set_title('State Assignment Accuracy', fontsize=font)

    res = [[i for i, j in bern_unpruned], [j for i, j in bern_unpruned]]
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

    res = [[i for i, j in gamma_a_unpruned], [j for i, j in gamma_a_unpruned]]
    ax[2].set_xlim((0, int(np.ceil(1.1 * max(x)))))
    ax[2].set_xlabel('Number of Lineages', fontsize=font2)
    ax[2].scatter(x, res[0], c='b', marker="o", label='Susceptible Unpruned', alpha=0.5)
    ax[2].scatter(x, res[1], c='r', marker="o", label='Resistant Unpruned', alpha=0.5)
    ax[2].set_ylabel(r'Gamma a $\beta$', rotation=90, fontsize=font2)
    ax[2].axhline(y=20, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b', alpha=0.6)
    ax[2].axhline(y=80, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[2].set_title('Gamma', fontsize=font)
    ax[2].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[2].legend(loc='best', framealpha=0.3)

    res = [[i for i, j in gamma_scale_unpruned], [j for i, j in gamma_scale_unpruned]]
    ax[3].set_xlim((0, int(np.ceil(1.1 * max(x)))))
    ax[3].set_xlabel('Number of Lineages', fontsize=font2)
    ax[3].scatter(x, res[0], c='b', marker="o", label='Susceptible Unpruned', alpha=0.5)
    ax[3].scatter(x, res[1], c='r', marker="o", label='Resistant Unpruned', alpha=0.5)
    ax[3].set_ylabel(r'Gamma scale $\alpha$', rotation=90, fontsize=font2)
    ax[3].axhline(y=20, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b', alpha=0.6)
    ax[3].axhline(y=80, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[3].set_title('Gamma', fontsize=font)
    ax[3].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[3].legend(loc='best', framealpha=0.3)

    ax[4].set_xlim((0, int(np.ceil(1.1 * max(x)))))
    ax[4].set_xlabel('Number of Lineages', fontsize=font2)
    ax[4].set_ylim(0, 110)
    ax[4].scatter(x, prunedNewAcc, c='k', marker="o", label='Accuracy', alpha=0.3)
    ax[4].axhline(y=100, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='k', alpha=0.6)  # linestyle is dashdotdotted
    ax[4].set_ylabel(r'Accuracy (\%)', rotation=90, fontsize=font2)
    ax[4].get_yticks()
    ax[4].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[4].set_title('State Assignment Accuracy', fontsize=font)

    res = [[i for i, j in bern_pruned], [j for i, j in bern_pruned]]
    ax[5].set_xlim((0, int(np.ceil(1.1 * max(x)))))
    ax[5].set_xlabel('Number of Lineages', fontsize=font2)
    ax[5].scatter(x, res[0], c='b', marker="o", label='Susceptible Unpruned', alpha=0.5)
    ax[5].scatter(x, res[1], c='r', marker="o", label='Resistant Unpruned', alpha=0.5)
    ax[5].set_ylabel('Bern $p$', rotation=90, fontsize=font2)
    ax[5].axhline(y=0.95, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b', alpha=0.6)
    ax[5].axhline(y=0.85, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[5].set_title('Bernoulli', fontsize=font)
    ax[5].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[5].legend(loc='best', framealpha=0.3)

    res = [[i for i, j in gamma_a_pruned], [j for i, j in gamma_a_pruned]]
    ax[6].set_xlim((0, int(np.ceil(1.1 * max(x)))))
    ax[6].set_xlabel('Number of Lineages', fontsize=font2)
    ax[6].scatter(x, res[0], c='b', marker="o", label='Susceptible Unpruned', alpha=0.5)
    ax[6].scatter(x, res[1], c='r', marker="o", label='Resistant Unpruned', alpha=0.5)
    ax[6].set_ylabel(r'Gamma a $\beta$', rotation=90, fontsize=font2)
    ax[6].axhline(y=20, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b', alpha=0.6)
    ax[6].axhline(y=80, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[6].set_title('Gamma', fontsize=font)
    ax[6].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[6].legend(loc='best', framealpha=0.3)

    res = [[i for i, j in gamma_scale_pruned], [j for i, j in gamma_scale_pruned]]
    ax[7].set_xlim((0, int(np.ceil(1.1 * max(x)))))
    ax[7].set_xlabel('Number of Lineages', fontsize=font2)
    ax[7].scatter(x, res[0], c='b', marker="o", label='Susceptible Unpruned', alpha=0.5)
    ax[7].scatter(x, res[1], c='r', marker="o", label='Resistant Unpruned', alpha=0.5)
    ax[7].set_ylabel(r'Gamma scale $\alpha$', rotation=90, fontsize=font2)
    ax[7].axhline(y=20, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b', alpha=0.6)
    ax[7].axhline(y=80, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[7].set_title('Gamma', fontsize=font)
    ax[7].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[7].legend(loc='best', framealpha=0.3)
