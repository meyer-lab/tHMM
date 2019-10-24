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

def KL_analyze():
    """ Assuming we have 2-state model """
    pi = np.array([0.6, 0.4], dtype="float")

    # T: transition probability matrix
    T = np.array([[0.65, 0.35],
                  [0.35, 0.65]])

    gamma_a0List = [5.0, 10.0, 15.0, 12.0]
    gamma_scale0List = [2.0, 2.0, 2.0, 3.3]
    gamma_a1List = [23.0, 20.0, 17.0, 12.0]
    gamma_scale1List = [3.0, 3.0, 3.0, 3.3]

    gammaKL = []
    acc = []
    for i in range(4):
        state_obj0 = StateDistribution(state0, bern_p0, gamma_a0List[i], gamma_loc, gamma_scale0List[i])
        state_obj1 = StateDistribution(state1, bern_p1, gamma_a1List[i], gamma_loc,  gamma_scale1List[i])

        E = [state_obj0, state_obj1]
        lineageObj = LineageTree(pi, T, E, desired_num_cells=2**11 -1, desired_experiment_time=100000, prune_condition='both', prune_boolean=False)
        X = [lineageObj]
        states = [cell.state for cell in lineageObj.output_lineage]
        deltas, state_ptrs, all_states, tHMMobj, NF, LL = Analyze(X, 2)

        # find the accuracy
        temp = accuracy(tHMMobj, all_states)
        acc.append(temp[0])

        # find the KL divergence
        state0obs=[]
        state1obs=[]

        for indx, cell in enumerate(lineageObj.output_lineage):
            if all_states[0][indx] == 0:
                state0obs.append(cell.obs[1])
            elif all_states[0][indx] == 1:
                state1obs.append(cell.obs[1])
        p=scipy.stats.gamma.pdf(state0obs, a=gamma_a0List[i], loc=gamma_loc, scale=gamma_scale0List[i])
        q=scipy.stats.gamma.pdf(state1obs, a=gamma_a1List[i], loc=gamma_loc, scale=gamma_scale1List[i])

        size = min(p.shape[0], q.shape[0])
        if size == 0:
            raise ValueError('the number of cells predicted in one of the states is zero! ')
        else:
            pprime = random.sample(list(p), size)
            qprime = random.sample(list(q), size)
        gammaKL.append(kl_divergence(np.asarray(pprime), np.asarray(qprime)))

    return KLgamma
    
        
        
        
        
    
