'''Calls the tHMM functions and outputs the parameters needed to generate the Figures'''
import copy as cp
import numpy as np
from .BaumWelch import fit
from .Viterbi import get_leaf_deltas, get_nonleaf_deltas, Viterbi
from .UpwardRecursion import get_leaf_Normalizing_Factors, get_leaf_betas, get_nonleaf_NF_and_betas, calculate_log_likelihood
from .tHMM import tHMM
from .StateDistribution import get_experiment_time


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


def accuracy(tHMMobj, all_states):
    acuracy_holder = []
    for num, lineageObj in enumerate(tHMMobj.X):
        lin_true_states = [cell.state for cell in lineageObj.output_lineage]

        bern_diff = np.zeros((lineageObj.num_states))
        gamma_a_diff = np.zeros((lineageObj.num_states))
        gamma_scale_diff = np.zeros((lineageObj.num_states))
        for state in range(lineageObj.num_states):
            bern_diff[state] = abs(tHMMobj.estimate.E[state].bern_p - lineageObj.E[0].bern_p)
            gamma_a_diff[state] = abs(tHMMobj.estimate.E[state].gamma_a - lineageObj.E[0].gamma_a)
            gamma_scale_diff[state] = abs(tHMMobj.estimate.E[state].gamma_scale - lineageObj.E[0].gamma_scale)

        bern_diff = bern_diff / sum(bern_diff)
        gamma_a_diff = gamma_a_diff / sum(gamma_a_diff)
        gamma_scale_diff = gamma_scale_diff / sum(gamma_scale_diff)

        total_errs = bern_diff + gamma_a_diff + gamma_scale_diff
        if total_errs[0] <= total_errs[1]:
            new_all_states = all_states[num]
        else:
            new_all_states = [not(x) for x in all_states[num]]
            tmp = cp.deepcopy(tHMMobj.estimate.E[1])
            tHMMobj.estimate.E[1] = tHMMobj.estimate.E[0]
            tHMMobj.estimate.E[0] = tmp

        counter = [1 if a == b else 0 for (a, b) in zip(new_all_states, lin_true_states)]
        acc = sum(counter) / len(lin_true_states)
        acuracy_holder.append(100 * acc)

    return acuracy_holder


# -------------------- when we have G1 and G2
def accuracyG(tHMMobj, all_states):
    acuracy_holder = []
    for num, lineageObj in enumerate(tHMMobj.X):
        lin_true_states = [cell.state for cell in lineageObj.output_lineage]

        bern_diff = np.zeros((lineageObj.num_states))
        gamma_aG1_diff = np.zeros((lineageObj.num_states))
        gamma_scaleG1_diff = np.zeros((lineageObj.num_states))
        gamma_aG2_diff = np.zeros((lineageObj.num_states))
        gamma_scaleG2_diff = np.zeros((lineageObj.num_states))
        for state in range(lineageObj.num_states):
            bern_diff[state] = abs(tHMMobj.estimate.E[state].bern_p - lineageObj.E[0].bern_p)
            gamma_aG1_diff[state] = abs(tHMMobj.estimate.E[state].gamma_a1 - lineageObj.E[0].gamma_a1)
            gamma_scaleG1_diff[state] = abs(tHMMobj.estimate.E[state].gamma_scale1 - lineageObj.E[0].gamma_scale1)
            gamma_aG2_diff[state] = abs(tHMMobj.estimate.E[state].gamma_a2 - lineageObj.E[0].gamma_a2)
            gamma_scaleG2_diff[state] = abs(tHMMobj.estimate.E[state].gamma_scale2 - lineageObj.E[0].gamma_scale2)

        bern_diff = bern_diff / sum(bern_diff)
        gamma_aG1_diff = gamma_aG1_diff / sum(gamma_aG1_diff)
        gamma_scaleG1_diff = gamma_scaleG1_diff / sum(gamma_scaleG1_diff)
        gamma_aG2_diff = gamma_aG2_diff / sum(gamma_aG2_diff)
        gamma_scaleG2_diff = gamma_scaleG2_diff / sum(gamma_scaleG2_diff)

        total_errs = bern_diff + gamma_aG1_diff + gamma_scaleG1_diff + gamma_aG2_diff + gamma_scaleG2_diff
        if total_errs[0] <= total_errs[1]:
            new_all_states = all_states[num]
        else:
            new_all_states = [not(x) for x in all_states[num]]
            tmp = cp.deepcopy(tHMMobj.estimate.E[1])
            tHMMobj.estimate.E[1] = tHMMobj.estimate.E[0]
            tHMMobj.estimate.E[0] = tmp

        counter = [1 if a == b else 0 for (a, b) in zip(new_all_states, lin_true_states)]
        acc = sum(counter) / len(lin_true_states)
        acuracy_holder.append(100 * acc)

    return acuracy_holder

def kl_divergence(p, q):
    """ Performs KL-divergence as:
        KL(P||Q) = Integral[ P(x) log(P(x)/Q(x)) ] for continuous distributions,
        and summation instead of integral, for discrete distributions. """
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def KL_analyze(lineageObj):
    """ Assuming we have 2-state model """

    obs0 = lineageObj.lineage_stats[0].full_lin_cells_obs
    obs1 = lineageObj.lineage_stats[1].full_lin_cells_obs
    obs0 = list(zip(*obs0))
    bern0 = np.asarray(obs0[0])
    lifetime0 = np.asarray(obs0[1])

    obs1 = list(zip(*obs1))
    bern1 = np.asarray(obs1[0])
    lifetime1 = np.asarray(obs1[1])

    size = min(lifetime0.shape, lifetime1.shape)
    KL_bern = kl_divergence(bern0[:size[0]], bern1[:size[0]])
    KL_gamma = kl_divergence(lifetime0[:size[0]], lifetime1[:size[0]])
    return KL_bern, KL_gamma
    
        
        
        
        
    
