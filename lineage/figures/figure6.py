"""
This creates Figure 6.
"""
from .figureCommon import getSetup
from ..Analyze import accuracy, Analyze
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
<<<<<<< HEAD
    ax, f = getSetup((30, 10), (2, 6))

    x_unpruned, accuracies_unpruned, bern_unpruned, bern_p0, bern_p1, gamma_a_unpruned, gamma_a0, gamma_a1, gamma_scale_unpruned, gamma_scale0, gamma_scale1, x_pruned, accuracies_pruned, bern_pruned, gamma_a_pruned, gamma_scale_pruned, tr_unprunedNorm, tr_prunedNorm, pi_unprunedNorm, pi_prunedNorm = accuracy_increased_cells()
    figure_maker(ax, x_unpruned, accuracies_unpruned, bern_unpruned, bern_p0, bern_p1, gamma_a_unpruned, gamma_a0, gamma_a1, gamma_scale_unpruned, gamma_scale0, gamma_scale1, x_pruned, accuracies_pruned, bern_pruned, gamma_a_pruned, gamma_scale_pruned, tr_unprunedNorm, tr_prunedNorm, pi_unprunedNorm, pi_prunedNorm)
=======
    ax, f = getSetup((20, 10), (2, 5))


#     x_unpruned, accuracies_unpruned, bern_unpruned, bern_p0, bern_p1, gamma_a_unpruned, gamma_a0, gamma_a1, gamma_scale_unpruned, gamma_scale0, gamma_scale1, x_pruned, accuracies_pruned, bern_pruned, gamma_a_pruned, gamma_scale_pruned = accuracy_increased_cells()
#     figure_maker(ax, x_unpruned, accuracies_unpruned, bern_unpruned, bern_p0, bern_p1, gamma_a_unpruned, gamma_a0, gamma_a1, gamma_scale_unpruned, gamma_scale0, gamma_scale1, x_pruned, accuracies_pruned, bern_pruned, gamma_a_pruned, gamma_scale_pruned)
    desired_num_cells, accuracies_unpruned, bern_unpruned, bern_p0, bern_p1, gamma_aG1_unpruned, gamma_aG11, gamma_aG12, gamma_aG21, gamma_aG22, gamma_scaleG1_unpruned, gamma_scaleG2_unpruned, gamma_scaleG11, gamma_scaleG12, gamma_scaleG21, gamma_scaleG22, accuracies_pruned, bern_pruned, gamma_aG1_pruned, gamma_scaleG1_pruned, gamma_aG2_pruned, gamma_scaleG2_pruned = accuracy_increased_cellsG()

>>>>>>> cbce299fd0200548e376af0e9b8cb07bde647b0b

    f.tight_layout()
    return f


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
    gamma_scale0 = 5

    # State 1 parameters "Susceptible"
    state1 = 1
    bern_p1 = 0.88
    gamma_a1 = 10
    gamma_scale1 = 1

    state_obj0 = StateDistribution(state0, bern_p0, gamma_a0, gamma_scale0)
    state_obj1 = StateDistribution(state1, bern_p1, gamma_a1, gamma_scale1)
    E = [state_obj0, state_obj1]
    
    desired_num_cells = np.logspace(5, 10, num=2, base=2.0)
    desired_num_cells = [num_cell - 1 for num_cell in desired_num_cells]
    
    x_unpruned = []
    x_pruned = []
    accuracies_unpruned = []
    accuracies_pruned = []
    bern_unpruned = []
    gamma_a_unpruned = []
    gamma_scale_unpruned = []
    bern_pruned = []
    gamma_a_pruned = []
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
        deltas, state_ptrs, all_states, tHMMobj, NF, LL = Analyze(X1, 2)
        deltas2, state_ptrs2, all_states2, tHMMobj2, NF2, LL2 = Analyze(X2, 2)
        
        # Collecting the accuracies of the lineages
        acc1 = accuracy(tHMMobj, all_states)[0]
        acc2 = accuracy(tHMMobj2, all_states2)[0]
        accuracies_unpruned.append(acc1)
        accuracies_pruned.append(acc2)

        # Collecting the parameter estimations
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

    # Transition and Pi estimates
        transition_mat_unpruned = tHMMobj.estimate.T # unpruned
        transition_mat_pruned = tHMMobj2.estimate.T # pruned
        
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

<<<<<<< HEAD

def figure_maker(ax, x_unpruned, accuracies_unpruned, bern_unpruned, bern_p0, bern_p1, gamma_a_unpruned, gamma_a0, gamma_a1, gamma_scale_unpruned, gamma_scale0, gamma_scale1, x_pruned, accuracies_pruned, bern_pruned, gamma_a_pruned, gamma_scale_pruned, tr_unprunedNorm, tr_prunedNorm, pi_unprunedNorm, pi_prunedNorm):
=======
#----------- Figure 6 for G1G2

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
    gamma_aG11 = 5
    gamma_scaleG11 = 1.5
    gamma_aG21 = 10
    gamma_scaleG21 = 1.5

    # State 1 parameters "Susceptible"
    state1 = 1
    bern_p1 = 0.88
    gamma_aG12 = 8
    gamma_scaleG12 = 1.0
    gamma_aG22 = 16
    gamma_scaleG22 = 1.0

    state_obj0 = StateDistribution(state0, bern_p0, gamma_aG11, gamma_scaleG11, gamma_aG21, gamma_scaleG21)
    state_obj1 = StateDistribution(state1, bern_p1, gamma_aG12, gamma_scaleG12, gamma_aG22, gamma_scaleG22)

    E = [state_obj0, state_obj1]
    # the key part in this function
    desired_num_cells = np.logspace(8, 10, num=2, base=2.0)
    desired_num_cells = [num_cell - 1 for num_cell in desired_num_cells]

    accuracies_unpruned = []
    accuracies_pruned = []
    bern_unpruned = []
    gamma_aG1_unpruned = []
    gamma_scaleG1_unpruned = []
    gamma_aG2_unpruned = []
    gamma_scaleG2_unpruned = []
    bern_pruned = []
    gamma_aG1_pruned = []
    gamma_scaleG1_pruned = []
    gamma_aG2_pruned = []
    gamma_scaleG2_pruned = []

    for num in desired_num_cells:
        lineage_unpruned = LineageTree(pi, T, E, num, prune_boolean=False)
        lineage_pruned = cp.deepcopy(lineage_unpruned)
        lineage_pruned.prune_boolean = True

        X1 = [lineage_unpruned]
        X2 = [lineage_pruned]
        print("unpruned")
        deltas, state_ptrs, all_states, tHMMobj, NF, LL = Analyze(X1, 2)
        deltas2, state_ptrs2, all_states2, tHMMobj2, NF2, LL2 = Analyze(X2, 2)
        acc1 = accuracyG(tHMMobj, all_states)[0]
        acc2 = accuracyG(tHMMobj2, all_states2)[0]
        accuracies_unpruned.append(acc1)
        accuracies_pruned.append(acc2)

        bern_p_total = ()
        gamma_aG1_total = ()
        gamma_scaleG1_total = ()
        gamma_aG2_total = ()
        gamma_scaleG2_total = ()
        bern_p_total2 = ()
        gamma_aG1_total2 = ()
        gamma_scaleG1_total2 = ()
        gamma_aG2_total2 = ()
        gamma_scaleG2_total2 = ()
        for state in range(tHMMobj.numStates):
            bern_p_total += (tHMMobj.estimate.E[state].bern_p,)
            gamma_aG1_total += (tHMMobj.estimate.E[state].gamma_aG1,)
            gamma_scaleG1_total += (tHMMobj.estimate.E[state].gamma_scaleG1,)
            gamma_aG2_total += (tHMMobj.estimate.E[state].gamma_aG2,)
            gamma_scaleG2_total += (tHMMobj.estimate.E[state].gamma_scaleG2,)

            bern_p_total2 += (tHMMobj2.estimate.E[state].bern_p,)
            gamma_aG1_total2 += (tHMMobj2.estimate.E[state].gamma_aG1,)
            gamma_scaleG1_total2 += (tHMMobj2.estimate.E[state].gamma_scaleG1,)
            gamma_aG2_total2 += (tHMMobj2.estimate.E[state].gamma_aG2,)
            gamma_scaleG2_total2 += (tHMMobj2.estimate.E[state].gamma_scaleG2,)

        bern_unpruned.append(bern_p_total)
        gamma_aG1_unpruned.append(gamma_aG1_total)
        gamma_scaleG1_unpruned.append(gamma_scaleG1_total)
        gamma_aG2_unpruned.append(gamma_aG2_total)
        gamma_scaleG2_unpruned.append(gamma_scaleG2_total)

        bern_pruned.append(bern_p_total2)
        gamma_aG1_pruned.append(gamma_aG1_total2)
        gamma_scaleG1_pruned.append(gamma_scaleG1_total2)
        gamma_aG2_pruned.append(gamma_aG2_total2)
        gamma_scaleG2_pruned.append(gamma_scaleG2_total2)

    return desired_num_cells, accuracies_unpruned, bern_unpruned, bern_p0, bern_p1, gamma_aG1_unpruned, gamma_aG11, gamma_aG12, gamma_aG21, gamma_aG22, gamma_scaleG1_unpruned, gamma_scaleG2_unpruned, gamma_scaleG11, gamma_scaleG12, gamma_scaleG21, gamma_scaleG22, accuracies_pruned, bern_pruned, gamma_aG1_pruned, gamma_scaleG1_pruned, gamma_aG2_pruned, gamma_scaleG2_pruned



def figure_maker(ax, x_unpruned, accuracies_unpruned, bern_unpruned, bern_p0, bern_p1, gamma_a_unpruned, gamma_a0, gamma_a1, gamma_scale_unpruned, gamma_scale0, gamma_scale1, x_pruned, accuracies_pruned, bern_pruned, gamma_a_pruned, gamma_scale_pruned):
>>>>>>> cbce299fd0200548e376af0e9b8cb07bde647b0b

    font = 11
    font2 = 10
    i=0
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x_unpruned)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)
    ax[i].set_ylim(0, 110)
    ax[i].scatter(x_unpruned, accuracies_unpruned, c='k', marker="o", label='Accuracy', alpha=0.3)
    ax[i].axhline(y=100, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='k', alpha=0.6)  # linestyle is dashdotdotted
    ax[i].set_ylabel(r'Accuracy (\%)', rotation=90, fontsize=font2)
    ax[i].get_yticks()
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].set_title('State Assignment Accuracy', fontsize=font)

    i+=1
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

    i+=1
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

    i+=1
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

    i+=1
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x_unpruned)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)
    ax[i].scatter(x_unpruned, tr_unprunedNorm, c='k', marker="o", label=' Unpruned', alpha=0.5)
    ax[i].set_ylabel(r'$||T-T_{est}||_{F}$', rotation=90, fontsize=font2)
    ax[i].axhline(y=0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='k', alpha=0.6)
    ax[i].set_title('Norm Transition', fontsize=font)
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].legend(loc='best', framealpha=0.3)

    i+=1
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x_unpruned)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)
    ax[i].scatter(x_unpruned, pi_unprunedNorm, c='k', marker="o", label=' Unpruned', alpha=0.5)
    ax[i].set_ylabel(r'$||\pi-\pi_{est}||_{2}$', rotation=90, fontsize=font2)
    ax[i].axhline(y=0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='k', alpha=0.6)
    ax[i].set_title('Norm Pi', fontsize=font)
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].legend(loc='best', framealpha=0.3)

    i+=1
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x_unpruned)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)
    ax[i].set_ylim(0, 110)
    ax[i].scatter(x_pruned, accuracies_pruned, c='k', marker="o", label='Accuracy', alpha=0.3)
    ax[i].axhline(y=100, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='k', alpha=0.6)  # linestyle is dashdotdotted
    ax[i].set_ylabel(r'Accuracy (\%)', rotation=90, fontsize=font2)
    ax[i].get_yticks()
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].set_title('State Assignment Accuracy', fontsize=font)

    i+=1
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

    i+=1
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

    i+=1
    res = [[i for i, j in gamma_scale_pruned], [j for i, j in gamma_scale_pruned]]
<<<<<<< HEAD
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

    i+=1
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x_pruned)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)
    ax[i].scatter(x_pruned, tr_prunedNorm, c='k', marker="o", label=' Pruned', alpha=0.5)
    ax[i].set_ylabel(r'$||T-T_{est}||_{F}$', rotation=90, fontsize=font2)
    ax[i].axhline(y=0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='k', alpha=0.6)
    ax[i].set_title('Norm Transition', fontsize=font)
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].legend(loc='best', framealpha=0.3)

    i+=1
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(x_pruned)))))
    ax[i].set_xlabel('Number of Cells', fontsize=font2)
    ax[i].scatter(x_pruned, pi_prunedNorm, c='k', marker="o", label=' Pruned', alpha=0.5)
    ax[i].set_ylabel(r'$||\pi-\pi_{est}||_{2}$', rotation=90, fontsize=font2)
    ax[i].axhline(y=0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='k', alpha=0.6)
    ax[i].set_title('Norm Pi', fontsize=font)
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].legend(loc='best', framealpha=0.3)
=======
    ax[7].set_xlim((0, int(np.ceil(1.1 * max(x_unpruned)))))
    ax[7].set_xlabel('Number of Cells', fontsize=font2)
    ax[7].scatter(x_pruned, res[0], c='b', marker="o", label='Susceptible Pruned', alpha=0.5)
    ax[7].scatter(x_pruned, res[1], c='r', marker="o", label='Resistant Pruned', alpha=0.5)
    ax[7].set_ylabel(r'Gamma scale $\alpha$', rotation=90, fontsize=font2)
    ax[7].axhline(y=gamma_scale0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b', alpha=0.6)
    ax[7].axhline(y=gamma_scale1, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[7].set_title('Gamma', fontsize=font)
    ax[7].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[7].legend(loc='best', framealpha=0.3)


#------------- figure for G1G2

def figure_makerG(ax, desired_num_cells, accuracies_unpruned, bern_unpruned, bern_p0, bern_p1, gamma_aG1_unpruned, gamma_aG11, gamma_aG12, gamma_aG21, gamma_aG22, gamma_scaleG1_unpruned, gamma_scaleG2_unpruned, gamma_scaleG11, gamma_scaleG12, gamma_scaleG21, gamma_scaleG22, accuracies_pruned, bern_pruned, gamma_aG1_pruned, gamma_scaleG1_pruned, gamma_aG2_pruned, gamma_scaleG2_pruned):

    font = 11
    font2 = 10
    ax[0].set_xlim((0, int(np.ceil(1.1 * max(x_unpruned)))))
    ax[0].set_xlabel('Number of Cells', fontsize=font2)
    ax[0].set_ylim(0, 110)
    ax[0].scatter(x_unpruned, accuracies_unpruned, c='k', marker="o", label='Accuracy', alpha=0.3)
    ax[0].axhline(y=100, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='k', alpha=0.6)  # linestyle is dashdotdotted
    ax[0].set_ylabel(r'Accuracy (\%)', rotation=90, fontsize=font2)
    ax[0].get_yticks()
    ax[0].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[0].set_title('State Assignment Accuracy', fontsize=font)

    res = [[i for i, j in bern_unpruned], [j for i, j in bern_unpruned]]
    ax[1].set_xlim((0, int(np.ceil(1.1 * max(x_unpruned)))))
    ax[1].set_xlabel('Number of Cells', fontsize=font2)
    ax[1].scatter(x_unpruned, res[0], c='b', marker="o", label='Susceptible Unpruned', alpha=0.5)
    ax[1].scatter(x_unpruned, res[1], c='r', marker="o", label='Resistant Unpruned', alpha=0.5)
    ax[1].set_ylabel('Bern $p$', rotation=90, fontsize=font2)
    ax[1].axhline(y=bern_p0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b', alpha=0.6)
    ax[1].axhline(y=bern_p1, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[1].set_title('Bernoulli', fontsize=font)
    ax[1].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[1].legend(loc='best', framealpha=0.3)

    res = [[i for i, j in gamma_aG1_unpruned], [j for i, j in gamma_aG1_unpruned]]
    ax[2].set_xlim((0, int(np.ceil(1.1 * max(x_unpruned)))))
    ax[2].set_xlabel('Number of Cells', fontsize=font2)
    ax[2].scatter(x_unpruned, res[0], c='b', marker="o", label='Susceptible Unpruned', alpha=0.5)
    ax[2].scatter(x_unpruned, res[1], c='r', marker="o", label='Resistant Unpruned', alpha=0.5)
    ax[2].set_ylabel(r'Gamma a $\beta$', rotation=90, fontsize=font2)
    ax[2].axhline(y=gamma_a0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b', alpha=0.6)
    ax[2].axhline(y=gamma_a1, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[2].set_title('Gamma', fontsize=font)
    ax[2].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[2].legend(loc='best', framealpha=0.3)

    res = [[i for i, j in gamma_scale_unpruned], [j for i, j in gamma_scale_unpruned]]
    ax[3].set_xlim((0, int(np.ceil(1.1 * max(x_unpruned)))))
    ax[3].set_xlabel('Number of Cells', fontsize=font2)
    ax[3].scatter(x_unpruned, res[0], c='b', marker="o", label='Susceptible Unpruned', alpha=0.5)
    ax[3].scatter(x_unpruned, res[1], c='r', marker="o", label='Resistant Unpruned', alpha=0.5)
    ax[3].set_ylabel(r'Gamma scale $\alpha$', rotation=90, fontsize=font2)
    ax[3].axhline(y=gamma_scale0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b', alpha=0.6)
    ax[3].axhline(y=gamma_scale1, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[3].set_title('Gamma', fontsize=font)
    ax[3].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[3].legend(loc='best', framealpha=0.3)

    ax[4].set_xlim((0, int(np.ceil(1.1 * max(x_unpruned)))))
    ax[4].set_xlabel('Number of Cells', fontsize=font2)
    ax[4].set_ylim(0, 110)
    ax[4].scatter(x_pruned, accuracies_pruned, c='k', marker="o", label='Accuracy', alpha=0.3)
    ax[4].axhline(y=100, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='k', alpha=0.6)  # linestyle is dashdotdotted
    ax[4].set_ylabel(r'Accuracy (\%)', rotation=90, fontsize=font2)
    ax[4].get_yticks()
    ax[4].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[4].set_title('State Assignment Accuracy', fontsize=font)

    res = [[i for i, j in bern_pruned], [j for i, j in bern_pruned]]
    ax[5].set_xlim((0, int(np.ceil(1.1 * max(x_unpruned)))))
    ax[5].set_xlabel('Number of Cells', fontsize=font2)
    ax[5].scatter(x_pruned, res[0], c='b', marker="o", label='Susceptible Pruned', alpha=0.5)
    ax[5].scatter(x_pruned, res[1], c='r', marker="o", label='Resistant Pruned', alpha=0.5)
    ax[5].set_ylabel('Bern $p$', rotation=90, fontsize=font2)
    ax[5].axhline(y=bern_p0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b', alpha=0.6)
    ax[5].axhline(y=bern_p1, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[5].set_title('Bernoulli', fontsize=font)
    ax[5].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[5].legend(loc='best', framealpha=0.3)

    res = [[i for i, j in gamma_a_pruned], [j for i, j in gamma_a_pruned]]
    ax[6].set_xlim((0, int(np.ceil(1.1 * max(x_unpruned)))))
    ax[6].set_xlabel('Number of Cells', fontsize=font2)
    ax[6].scatter(x_pruned, res[0], c='b', marker="o", label='Susceptible Pruned', alpha=0.5)
    ax[6].scatter(x_pruned, res[1], c='r', marker="o", label='Resistant Pruned', alpha=0.5)
    ax[6].set_ylabel(r'Gamma a $\beta$', rotation=90, fontsize=font2)
    ax[6].axhline(y=gamma_a0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b', alpha=0.6)
    ax[6].axhline(y=gamma_a1, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[6].set_title('Gamma', fontsize=font)
    ax[6].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[6].legend(loc='best', framealpha=0.3)

    res = [[i for i, j in gamma_scale_pruned], [j for i, j in gamma_scale_pruned]]
    ax[7].set_xlim((0, int(np.ceil(1.1 * max(x_unpruned)))))
    ax[7].set_xlabel('Number of Cells', fontsize=font2)
    ax[7].scatter(x_pruned, res[0], c='b', marker="o", label='Susceptible Pruned', alpha=0.5)
    ax[7].scatter(x_pruned, res[1], c='r', marker="o", label='Resistant Pruned', alpha=0.5)
    ax[7].set_ylabel(r'Gamma scale $\alpha$', rotation=90, fontsize=font2)
    ax[7].axhline(y=gamma_scale0, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b', alpha=0.6)
    ax[7].axhline(y=gamma_scale1, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[7].set_title('Gamma', fontsize=font)
    ax[7].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[7].legend(loc='best', framealpha=0.3)
>>>>>>> cbce299fd0200548e376af0e9b8cb07bde647b0b
