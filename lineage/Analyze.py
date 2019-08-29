'''Calls the tHMM functions and outputs the parameters needed to generate the Figures'''

from .BaumWelch import fit
from .Viterbi import get_leaf_deltas, get_nonleaf_deltas, Viterbi
from .UpwardRecursion import get_leaf_Normalizing_Factors, get_leaf_betas, get_nonleaf_NF_and_betas, calculate_log_likelihood
from .tHMM import tHMM
from .LineageTree import LineageTree
from .StateDistribution import StateDistribution


def Analyze(X, numStates):
    """Runs a tHMM and outputs state classification from viterbi, thmm object, normalizing factor, log likelihood, and deltas.
    Args:
    -----
    X {list}: A list containing LineageTree objects as lineages.
    numStates {Int}: The number of states we want our model to estimate for the given population.

    Returns:
    --------
    deltas {}:
    state_ptrs {}:
    all_states {}:
    tHMMobj {obj}:
    NF {vector}: A N x 1 matrix, each element is for each cell which is basically marginal observation distribution.
    LL {}:
    """

    tHMMobj = tHMM(X, numStates=numStates)  # build the tHMM class with X
    fit(tHMMobj, max_iter=200)

    deltas, state_ptrs = get_leaf_deltas(tHMMobj)  # gets the deltas matrix
    get_nonleaf_deltas(tHMMobj, deltas, state_ptrs)
    all_states = Viterbi(tHMMobj, deltas, state_ptrs)
    NF = get_leaf_Normalizing_Factors(tHMMobj)
    betas = get_leaf_betas(tHMMobj, NF)
    get_nonleaf_NF_and_betas(tHMMobj, NF, betas)
    LL = calculate_log_likelihood(tHMMobj, NF)
    return(deltas, state_ptrs, all_states, tHMMobj, NF, LL)

#


def accuracy(X, all_states):
    for num, lineageObj in enumerate(X):
        lin_estimated_states = all_states[num]
        lin_true_states = [cell.state for cell in lineageObj.output_lineage]
        total = len(lin_estimated_states)
        assert total == len(lin_true_states)
        counter = [1. if a == b else 0. for (a, b) in zip(lin_estimated_states, lin_true_states)]
        acc = sum(counter) / total
    return max(acc, 1 - acc)
