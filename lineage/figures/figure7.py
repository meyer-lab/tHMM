"""
This creates Figure 7. AIC Figure.
"""
from .figureCommon import getSetup


def makeFigure():
    # Get list of axis objects
    ax, f = getSetup((5, 5), (1, 1))
    

    f.tight_layout()

    return f

    
##-------------------- Figure 7   
def accuracy_increased_lineages():
    """ Calclates accuracy and parameter estimation by increasing the number of lineages. """
    # pi: the initial probability vector
    pi = np.array([0.5, 0.5], dtype="float")

    # T: transition probability matrix
    T = np.array([[0.99, 0.01],
              [0.15, 0.85]])

    # State 0 parameters "Resistant"
    state0 = 0
    bern_p0 = 0.95
    gamma_a0 = 5.0
    gamma_scale0 = 1.0

    # State 1 parameters "Susciptible"
    state1 = 1
    bern_p1 = 0.8
    gamma_a1 = 10.0
    gamma_scale1 = 2.0

    state_obj0 = StateDistribution(state0, bern_p0, gamma_a0, gamma_scale0)
    state_obj1 = StateDistribution(state1, bern_p1, gamma_a1, gamma_scale1)

    E = [state_obj0, state_obj1]

    # 127 cells in every lineage
    desired_num_cells = 2**7 - 1
    # increasing number of lineages from 1 to 10 and calculating accuracy and estimate parameters for both pruned and unpruned lineages.
    num_lineages = list(range(1, 10))

    accuracies_unpruned = []
    accuracies_pruned = []
    bern_unpruned = []
    gamma_a_unpruned = []
    gamma_b_unpruned = []
    bern_pruned = []
    gamma_a_pruned = []
    gamma_b_pruned = []

    X_p = []
    X_u = []
    for num in num_lineages:
        # unpruned lineage
        lineage_unpruned = LineageTree(pi, T, E, desired_num_cells, prune_boolean=False)
        # pruned lineage
        lineage_pruned = lineage_unpruned.prune_boolean(True)

        X_p.append(lineage_unpruned)
        X_u.append(lineage_pruned)
        deltas, state_ptrs, all_states, tHMMobj, NF, LL = Analyze(X_p, 2) 
        deltas2, state_ptrs2, all_states2, tHMMobj2, NF2, LL2 = Analyze(X_u, 2) 
        acc1 = accuracy(X_p, all_states)
        acc2 = accuracy(X_u, all_states2)
        accuracies_unpruned.append(100*acc1)        
        accuracies_pruned.append(100*acc2)
        

        # unpruned lineage

        bern_p_total = []
        gamma_a_total = []
        gamma_b_total = []
        for state in range(tHMMobj.numStates):
            bern_p_estimate = tHMMobj.estimate.E[state].bern_p
            gamma_a_estimate = tHMMobj.estimate.E[state].gamma_a
            gamma_b_estimate = tHMMobj.estimate.E[state].gamma_scale
            bern_p_total.append(bern_p_estimate)
            gamma_a_total.append(gamma_a_estimate)
            gamma_b_total.append(gamma_b_estimate)
        bern_unpruned.append(bern_p_total)
        gamma_a_unpruned.append(gamma_a_total)
        gamma_b_unpruned.append(gamma_b_total)

        # pruned lineage

        bern_p_total_p = []
        gamma_a_total_p = []
        gamma_b_total_p = []
        for state in range(tHMMobj2.numStates):
            bern_p_estimate_p = tHMMobj2.estimate.E[state].bern_p
            gamma_a_estimate_p = tHMMobj2.estimate.E[state].gamma_a
            gamma_b_estimate_p = tHMMobj2.estimate.E[state].gamma_scale
            bern_p_total_p.append(bern_p_estimate_p)
            gamma_a_total_p.append(gamma_a_estimate_p)
            gamma_b_total_p.append(gamma_b_estimate_p)
        bern_pruned.append(bern_p_total_p)
        gamma_a_pruned.append(gamma_a_total_p)
        gamma_b_pruned.append(gamma_b_total_p)
        
    return desired_num_cells, accuracies_unpruned, accuracies_pruned, bern_unpruned, gamma_a_unpruned, gamma_b_unpruned, bern_pruned, gamma_a_pruned, gamma_b_pruned


def figure_maker(ax, num_lineages, accuracies_unpruned, accuracies_pruned, bern_unpruned, gamma_a_unpruned, gamma_b_unpruned, bern_pruned, gamma_a_pruned, gamma_b_pruned):
    x = num_lineages
    font2 = 10
    ax[0].set_xlim((0, int(np.ceil(1.1 * max(x)))))
    ax[0].set_xlabel('Number of Lineages', fontsize=font2)
    ax[0].set_ylim(0, 110)
    ax[0].errorbar(x, acc_h1, fmt='o', c='k', marker="o", label='Accuracy', alpha=0.3)
    ax[0].plot(sorted_x_vs_acc[:, 0][9:], moving_average(sorted_x_vs_acc[:, 1]), c='k', label='Moving Average')
    ax[0].axhline(y=100, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='k', alpha=0.6)  # linestyle is dashdotdotted
    ax[0].set_ylabel('Accuracy (%)', rotation=90, fontsize=font2)
    ax[0].get_yticks()
    ax[0].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.4)
    ax[0].set_title('State Assignment Accuracy', fontsize=font)

    #ax = axs[0, 1]
    ax[1].set_xlim((0, int(np.ceil(1.1 * max(x)))))
    ax[1].set_xlabel(xlabel, fontsize=font2)
    ax[1].errorbar(x, bern_MAS_h1, fmt='o', c='b', marker="o", label='Susceptible', alpha=0.2)
    ax[1].errorbar(x, bern_2_h1, fmt='o', c='r', marker="o", label='Resistant', alpha=0.2)
    ax[1].set_ylabel('Theta', rotation=90, fontsize=font2)
    ax[1].axhline(y=MASlocBern, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b', alpha=0.6)
    ax[1].axhline(y=locBern2, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[1].get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[1].set_title('Bernoulli', fontsize=font)
    ax[1].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.3)
    ax[1].legend(loc='best', framealpha=0.3)

    #ax = axs[1, 0]
    ax[2].set_xlim((0, int(np.ceil(1.1 * max(x)))))
    ax[2].set_xlabel(xlabel, fontsize=font2)
    #ax[2].set_xscale("log", nonposx='clip')
    ax[2].errorbar(x, betaExp_MAS_h1, fmt='o', c='b', marker="o", label='Susceptible', alpha=0.2)
    ax[2].errorbar(x, betaExp_2_h1, fmt='o', c='r', marker="o", label='Resistant', alpha=0.2)
    ax[2].axhline(y=MASbeta, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b', alpha=0.6)
    ax[2].axhline(y=beta2, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[2].set_ylabel(panel_3_ylabel, rotation=90, fontsize=font2)
    ax[2].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[2].get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[2].set_title(panel_3_title, fontsize=font)
    ax[2].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.3)
    ax[2].legend(loc='best', framealpha=0.3)
        