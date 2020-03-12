'''Calls the tHMM functions and outputs the parameters needed to generate the Figures'''
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .BaumWelch import fit
from .Viterbi import get_leaf_deltas, get_nonleaf_deltas, Viterbi
from .UpwardRecursion import get_leaf_Normalizing_Factors, get_leaf_betas, get_nonleaf_NF_and_betas, calculate_log_likelihood
from .tHMM import tHMM
from sklearn import metrics
from scipy.stats import entropy


def preAnalyze(X, num_states):
    """Runs a tHMM and outputs state classification from viterbi, thmm object, normalizing factor, log likelihood, and deltas.
    Args:
    -----
    X {list}: A list containing LineageTree objects as lineages.
    numStates {Int}: The number of states we want our model to estimate for the given population.

    Returns:
    --------
    tHMMobj {obj}:
    """

    for num_tries in range(1, 5):
        try:
            tHMMobj = tHMM(X, numStates=num_states)  # build the tHMM class with X
            fit(tHMMobj, max_iter=300)
            break
        except AssertionError:
            if num_tries == 4:
                print("Caught AssertionError in fitting after multiple ({}) runs. Fitting is breaking after trying {} times. Consider inspecting the length of your lineages.".format(num_tries))
                raise

    deltas, state_ptrs = get_leaf_deltas(tHMMobj)
    get_nonleaf_deltas(tHMMobj, deltas, state_ptrs)
    pred_states_by_lineage = Viterbi(tHMMobj, deltas, state_ptrs)
    NF = get_leaf_Normalizing_Factors(tHMMobj)
    betas = get_leaf_betas(tHMMobj, NF)
    get_nonleaf_NF_and_betas(tHMMobj, NF, betas)
    LL = calculate_log_likelihood(NF)
    return tHMMobj, pred_states_by_lineage, LL


def Analyze(X, num_states):
    tHMMobj, pred_states_by_lineage, LL = preAnalyze(X, num_states)

    for _ in range(1, 5):
        tmp_tHMMobj, tmp_pred_states_by_lineage, tmp_LL = preAnalyze(X, num_states)
        if tmp_LL > LL:
            tHMMobj = tmp_tHMMobj
            pred_states_by_lineage = tmp_pred_states_by_lineage
            LL = tmp_LL

    return tHMMobj, pred_states_by_lineage, LL


def run_Analyze_over(list_of_populations, num_states, parallel=True):
    """
    A function that can be parallelized to speed up figure creation.

    This function is the outermost for-loop we will end up using
    when analyzing heterogenous populations or lineages.

    Analyze is the bottleneck in the figure creation process. The
    rest of the code involved in figure creation deals with collecting
    and computing certain statistics, most of which can be done in an
    additional for loop over the results from Analyze.

    This function takes as input:
    list_of_populations: a list of populations that contain lineages
    num_states: an integer number of states to identify (a hyper-parameter of our model)
    """
    output = []
    if parallel:
        exe = ProcessPoolExecutor()

        prom_holder = []
        for _, population in enumerate(list_of_populations):
            prom_holder.append(exe.submit(Analyze, population, num_states))

        for _, prom in enumerate(prom_holder):
            output.append(prom.result())
    else:
        for _, population in enumerate(list_of_populations):
            output.append(Analyze(population, num_states))

    return output


def Results(tHMMobj, pred_states_by_lineage, LL):
    """
    This function calculates several results of fitting a synthetic lineage.
    """
    # Instantiating a dictionary to hold the various metrics of accuracy and scoring for the results of our method
    results_dict = {}
    results_dict["total_number_of_lineages"] = len(tHMMobj.X)
    results_dict["LL"] = LL

    # Calculate the predicted states prior to switching their label
    true_states = [cell.state for lineage_obj in tHMMobj.X for cell in lineage_obj.output_lineage]
    pred_states = [state for sublist in pred_states_by_lineage for state in sublist]

    results_dict["total_number_of_cells"] = len(pred_states)

    # 1. Calculate some cluster labeling scores between the true states and the predicted states prior to switching the
    # predicted state labels based on their underlying distributions

    # 1.1. mutual information score
    results_dict["mutual_info_score"] = metrics.mutual_info_score(true_states, pred_states)

    # 1.2. normalized mutual information score
    results_dict["mutual_info_score"] = metrics.normalized_mutual_info_score(true_states, pred_states)

    # 1.3. adjusted mutual information score
    results_dict["adjusted_mutual_info_score"] = metrics.adjusted_mutual_info_score(true_states, pred_states)

    # 1.4. adjusted Rand index
    results_dict["adjusted_rand_score"] = metrics.adjusted_rand_score(true_states, pred_states)

    # 1.5. V-measure cluster labeling score
    results_dict["v_measure_score"] = metrics.v_measure_score(true_states, pred_states)

    # 1.6. homogeneity metric
    results_dict["homogeneity_score"] = metrics.homogeneity_score(true_states, pred_states)

    # 1.7. completeness metric
    results_dict["completeness_score"] = metrics.completeness_score(true_states, pred_states)

    # 2. Switch the underlying state labels based on the KL-divergence of the underlying states' distributions

    # First collect all the observations from the entire population across the lineages ordered by state
    obs_by_state = []
    for state in range(tHMMobj.numStates):
        obs_by_state.append([obs for lineage in tHMMobj.X for obs in lineage.lineage_stats[state].full_lin_cells_obs])

    # Array to hold divergence values
    switcher_array = np.zeros((tHMMobj.numStates, tHMMobj.numStates), dtype="float")

    for state_pred in range(tHMMobj.numStates):
        for state_true in range(tHMMobj.numStates):
            p = [tHMMobj.estimate.E[state_pred].pdf(y) for y in obs_by_state[state_pred]]
            q = [tHMMobj.X[0].E[state_true].pdf(x) for x in obs_by_state[state_pred]]
            switcher_array[state_pred, state_true] = (entropy(p, q) + entropy(q, p))

    results_dict["switcher_array"] = switcher_array

    # Create switcher map based on the minimal entropies in the switcher array

    switcher_map = [None] * tHMMobj.numStates

    for row in range(tHMMobj.numStates):
        switcher_row = list(switcher_array[row, :])
        switcher_map[row] = switcher_row.index(min(switcher_row))

    results_dict["switcher_map"] = switcher_map

    # Rearrange the values in the transition matrix
    temp_T = tHMMobj.estimate.T
    for row_idx in range(tHMMobj.numStates):
        for col_idx in range(tHMMobj.numStates):
            temp_T[row_idx, col_idx] = tHMMobj.estimate.T[switcher_map[row_idx], switcher_map[col_idx]]

    results_dict["switched_transition_matrix"] = temp_T
    results_dict["transition_matrix_norm"] = np.linalg.norm(temp_T - tHMMobj.X[0].T)

    # Rearrange the values in the pi vector
    temp_pi = tHMMobj.estimate.pi
    for val_idx in range(tHMMobj.numStates):
        temp_pi[val_idx] = tHMMobj.estimate.pi[switcher_map[val_idx]]

    results_dict["switched_pi_vector"] = temp_pi
    results_dict["pi_vector_norm"] = np.linalg.norm(temp_pi - tHMMobj.X[0].pi)

    # Rearrange the emissions list
    temp_emissions = [None] * tHMMobj.numStates
    for val_idx in range(tHMMobj.numStates):
        temp_emissions[val_idx] = tHMMobj.estimate.E[switcher_map[val_idx]]

    results_dict["switched_emissions"] = temp_emissions

    # Get the estimated parameter values
    results_dict["param_estimates"] = []
    for val_idx in range(tHMMobj.numStates):
        results_dict["param_estimates"].append(temp_emissions[val_idx].params)

    # 3. Calculate accuracy after switching states
    pred_states_switched = [switcher_map[state] for state in pred_states]
    results_dict["state_counter"] = np.bincount(pred_states_switched)
    results_dict["state_proportions"] = [i/len(pred_states_switched) for i in results_dict["state_counter"]]
    results_dict["state_proportions_0"] = results_dict["state_proportions"][0]
    results_dict["accuracy_before_switching"] = 100 * sum([int(i == j) for i, j in zip(pred_states, true_states)]) / len(true_states)
    results_dict["accuracy_after_switching"] = 100 * sum([int(i == j) for i, j in zip(pred_states_switched, true_states)]) / len(true_states)

    return results_dict


def run_Results_over(output):
    """
    A function that can be parallelized to speed up figure creation
    This function takes as input:
    output: a list of tuples from the results of running run_Analyze_over
    """
    results_holder = []
    for output_idx, (tHMMobj, pred_states_by_lineage, LL) in enumerate(output):
        results_holder.append(Results(tHMMobj, pred_states_by_lineage, LL))

    return results_holder


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
        LL :
    Returns:
    --------
        AIC_value : containing AIC values relative to 0 for each lineage.
        AIC_degrees_of_freedom : the degrees of freedom in AIC calculation (numStates**2 + numStates * number_of_parameters - 1) - same for each lineage
    '''
    numStates = tHMMobj.numStates

    number_of_parameters = len(tHMMobj.estimate.E[0].params)
    AIC_degrees_of_freedom = numStates**2 + numStates * number_of_parameters - 1

    AIC_value = -2 * LL + 2 * AIC_degrees_of_freedom

    return AIC_value, AIC_degrees_of_freedom
