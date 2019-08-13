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