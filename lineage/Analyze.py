'''Calls the tHMM functions and outputs the parameters needed to generate the Figures'''
import copy as cp
import numpy as np
from .BaumWelch import fit
from .Viterbi import get_leaf_deltas, get_nonleaf_deltas, Viterbi
from .UpwardRecursion import get_leaf_Normalizing_Factors, get_leaf_betas, get_nonleaf_NF_and_betas, calculate_log_likelihood
from .tHMM import tHMM
from sklearn import metrics


def preAnalyze(X, numStates):
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

    for num_tries in range(1, 5):
        try:
            tHMMobj = tHMM(X, numStates=numStates)  # build the tHMM class with X
            fit(tHMMobj, max_iter=300)
            break
        except AssertionError:
            if num_tries == 4:
                print("Caught AssertionError in fitting after multiple ({}) runs. Fitting is breaking after trying {} times. Consider inspecting the length of your lineages.".format(num_tries))
                raise

    deltas, state_ptrs = get_leaf_deltas(tHMMobj)  # gets the deltas matrix
    get_nonleaf_deltas(tHMMobj, deltas, state_ptrs)
    pred_states_by_lineage = Viterbi(tHMMobj, deltas, state_ptrs)
    NF = get_leaf_Normalizing_Factors(tHMMobj)
    betas = get_leaf_betas(tHMMobj, NF)
    get_nonleaf_NF_and_betas(tHMMobj, NF, betas)
    LL = calculate_log_likelihood(tHMMobj, NF)
    return deltas, state_ptrs, pred_states_by_lineage, tHMMobj, NF, LL


def Analyze(X, numStates):
    deltas, state_ptrs, pred_states_by_lineage, tHMMobj, NF, LL = preAnalyze(X, numStates)

    for _ in range(1, 5):
        tmp_deltas, tmp_state_ptrs, tmp_pred_states_by_lineage, tmp_tHMMobj, tmp_NF, tmp_LL = preAnalyze(X, numStates)
        if tmp_LL > LL:
            deltas = tmp_deltas
            state_ptrs = tmp_state_ptrs
            pred_states_by_lineage = tmp_pred_states_by_lineage
            tHMMobj = tmp_tHMMobj
            NF = tmp_NF
            LL = tmp_LL
            

    return deltas, state_ptrs, pred_states_by_lineage, tHMMobj, NF, LL


def accuracy(tHMMobj, pred_states_by_lineage):
    """ 
    This function calculates the accuracy
    given estimated and true states.
    """
    ## Instantiating a dictionary to hold the various metrics of accuracy and scoring for the results of our method
    accuracies_dict = {}
    
    ## Calculate the predicted states prior to switching their label
    true_states = [cell.state for cell in lineage_obj.output_lineage for lineage_obj in tHMMobj.X]
    pred_states = [state for state in sublist for sublist in pred_states_by_lineage]
    
    ## 1. Calculate some cluster labeling scores between the true states and the predicted states prior to switching the 
    ## predicted state labels based on their underlying distributions
    
    # 1.1. mutual information score 
    accuracies_dict["mutual_info_score"] = metrics.mutual_info_score(true_states, pred_states)
    
    # 1.2. normalized mutual information score
    accuracies_dict["mutual_info_score"] = metrics.normalized_mutual_info_score(true_states, pred_states)
    
    # 1.3. adjusted mutual information score
    accuracies_dict["adjusted_mutual_info_score"] = metrics.adjusted_mutual_info_score(true_states, pred_states)
    
    # 1.4. adjusted Rand index
    accuracies_dict["adjusted_rand_score"] = metrics.adjusted_rand_score(true_states, pred_states)
    
    # 1.5. V-measure cluster labeling score
    accuracies_dict["v_measure_score"] = metrics.v_measure_score(true_states, pred_states)
    
    # 1.6. homogeneity metric
    accuracies_dict["homogeneity_score"] = metrics.homogeneity_score(true_states, pred_states)
    
    # 1.7. completeness metric
    accuracies_dict["completeness_score"] = metrics.completeness_score(true_states, pred_states)
    
    ## 2. Switch the underlying state labels based on the KL-divergence of the underlying states' distributions 
 
    return accuracies_dict


def get_stationary_distribution(transition_matrix):
    """
    Obtain the stationary distribution given a transition matrix.
    The transition matrix should be in the format preferred by Wikipedia.
    That is, the transition matrix should be square, contain real numbers,
    and be right stochastic.
    This implies that the rows of the transition matrix sum to one (not the columns).
    This also means that the row index (i) represents the state of the previous step,
    and the column index (j) represents the state of the next step.
    If the transition matrix is defined as A, then the element of the A matrix

    A[i,j] = P(child = j | parent = i).

    Our goal is to find the stationary distribution vector, p, such that

    pA = p.

    This is equivalent to finding the distribution of states that is invariant
    to the Markov transition operation.
    Remark. Due to our transition matrix being right stochastic, the stationary vector
    is a row-vector which is left-multiplying the transition matrix. Using the transpose
    can help with the notation.

    Notation:
    ().T is the transpose of ()
    * is the matrix multiplication operation
    I is the identity matrix of size K
    K() is a vector with K number of (), for example,
    K0 is a vector with K number of 0s
    K represents the number of states

    p*A             = p       1
    (p*A).T         = p.T     2
    A.T*p.T         = p.T     3
    A.T*p.T - I*p.T = K0      4
    (A.T - I)*p.T   = K0      5

    Our goal is to solve this equation. To obtain non-trivial solutions for p
    and to constrain the problem, we can add the constraint that the elements of
    p must sum to 1.

    [(A.T - I), K1]*p.T = [K0, 1]      6
    [(A.T - I), K1]     = B            7
    B*p.T               = [K0, 1]      8
    [K0, 1]             = c            9
    B*p.T               = c           10

    However, this is now an over-determined system of linear equations
    (B now has more rows than the number of elements (K) in p).
    Linear equation solvers will be unable to proceed.
    To ameliorate this, we can use the normal equations.

    B.T*B*p.T = B.T*c     11

    Solving this yields the stationary distribution vector.
    We can then check that the stationary distribution vector
    remains unchanged by applying the transition matrix to it
    and obtain the stationary distribution vector again.

    A.T*p.T=p.T  12

    We return this solution as a row vector.
    """
    A = transition_matrix
    K = A.shape[0]
    tmp_A = A.T - np.eye(K)  # 5
    B = np.r_[tmp_A, np.ones((1, K))]  # 7
    BT_B = np.matmul(B.T, B)
    c = np.zeros((K + 1, 1))  # 9
    c[K, 0] = 1
    p = np.linalg.solve(BT_B, np.matmul(B.T, c)).T  # 11
    assert np.allclose(np.matmul(transition_matrix.T, p.T), p.T)  # 12
    return p


def getAIC(tHMMobj, LL):
    '''
    Gets the AIC values. Akaike Information Criterion, used for model selection and deals with the trade off
    between over-fitting and under-fitting.
    AIC = 2*k - 2 * log(LL) in which k is the number of free parameters and LL is the maximum of likelihood function.
    Minimum of AIC detremines the relatively better model.
    Args:
    -----
        tHMMobj (obj): the tHMM class which has been built.
        LL (list): a list containing log-likelihood values of Normalizing Factors for each lineage.
    Returns:
    --------
        AIC_ls (list): containing AIC values relative to 0 for each lineage.
        LL_ls (list): containing LL values relative to 0 for each lineage.
        AIC_degrees_of_freedom : the degrees of freedom in AIC calculation (numStates**2 + numStates * number_of_parameters - 1) - same for each lineage
    '''
    numStates = tHMMobj.numStates

    number_of_parameters = len(tHMMobj.estimate.E[0].params)
    AIC_degrees_of_freedom = numStates**2 + numStates * number_of_parameters - 1

    AIC_value = -2 * LL + 2 * AIC_degrees_of_freedom

    return(AIC_value, AIC_degrees_of_freedom)  # no longer returning relative to zero


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
