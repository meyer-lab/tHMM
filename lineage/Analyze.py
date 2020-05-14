"""Calls the tHMM functions and outputs the parameters needed to generate the Figures"""
from concurrent.futures import ProcessPoolExecutor
import random
import numpy as np
from sklearn import metrics
from scipy.stats import entropy, wasserstein_distance
from .BaumWelch import fit
from .Viterbi import get_leaf_deltas, get_nonleaf_deltas, Viterbi
from .UpwardRecursion import get_leaf_Normalizing_Factors, get_leaf_betas, get_nonleaf_NF_and_betas, calculate_log_likelihood
from .tHMM import tHMM



def preAnalyze(X, num_states, max_iter=500, fpi=None, fT=None, fE=None):
    """Runs a tHMM and outputs state classification from viterbi, thmm object, normalizing factor, log likelihood, and deltas.
    Args:
    -----
    X {list}: A list containing LineageTree objects as lineages.
    num_states {Int}: The number of states we want our model to estimate for the given population.

    Returns:
    --------
    tHMMobj {obj}:
    """
    error_holder = []
    for num_tries in range(1, 15):
        try:
            tHMMobj = tHMM(X, num_states=num_states, fpi=fpi, fT=fT, fE=fE)  # build the tHMM class with X
            fit(tHMMobj, max_iter=max_iter)
            break
        except (AssertionError, ZeroDivisionError, RuntimeError) as error:
            error_holder.append(error)
            if num_tries == 14:
                print(
                    f"Caught the following errors: \n \n {error_holder} \n \n in fitting after multiple {num_tries} runs. Fitting is breaking after trying {num_tries} times. If you're facing a ZeroDivisionError or a RuntimeError then the most likely issue is the estimates of your parameters are returning nonsensible parameters. Consider changing your parameter estimator. "
                )
                raise

    deltas, state_ptrs = get_leaf_deltas(tHMMobj)
    get_nonleaf_deltas(tHMMobj, deltas, state_ptrs)
    pred_states_by_lineage = Viterbi(tHMMobj, deltas, state_ptrs)
    NF = get_leaf_Normalizing_Factors(tHMMobj)
    betas = get_leaf_betas(tHMMobj, NF)
    get_nonleaf_NF_and_betas(tHMMobj, NF, betas)
    LL = calculate_log_likelihood(NF)
    return tHMMobj, pred_states_by_lineage, LL


def Analyze(X, num_states, max_iter=500, fpi=None, fT=None, fE=None):
    """
    Analyze runs several for loops runnning our model for a given number of states
    given an input population (a list of lineages).
    """
    tHMMobj, pred_states_by_lineage, LL = preAnalyze(X, num_states, max_iter=max_iter, fpi=fpi, fT=fT, fE=fE)

    for _ in range(1, 5):
        tmp_tHMMobj, tmp_pred_states_by_lineage, tmp_LL = preAnalyze(X, num_states, max_iter=max_iter, fpi=fpi, fT=fT, fE=fE)
        if sum(tmp_LL) > sum(LL):
            tHMMobj = tmp_tHMMobj
            pred_states_by_lineage = tmp_pred_states_by_lineage
            LL = tmp_LL

    return tHMMobj, pred_states_by_lineage, LL


def run_Analyze_over(list_of_populations, num_states, parallel=True, **kwargs):
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
    list_of_fpi = kwargs.get('list_of_fpi', [None] * len(list_of_populations))
    list_of_fT = kwargs.get('list_of_fT', [None] * len(list_of_populations))
    list_of_fE = kwargs.get('list_of_fE', [None] * len(list_of_populations))
    output = []
    if parallel:
        exe = ProcessPoolExecutor()

        prom_holder = []
        for idx, population in enumerate(list_of_populations):
            prom_holder.append(exe.submit(Analyze, population, num_states, fpi=list_of_fpi[idx], fT=list_of_fT[idx], fE=list_of_fE[idx]))

        for _, prom in enumerate(prom_holder):
            output.append(prom.result())
    else:
        for idx, population in enumerate(list_of_populations):
            output.append(Analyze(population, num_states, fpi=list_of_fpi[idx], fT=list_of_fT[idx], fE=list_of_fE[idx]))

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
    for state in range(tHMMobj.num_states):
        obs_by_state.append([cell.obs for lineage in tHMMobj.X for cell in lineage.output_lineage if cell.state == state])

    # Array to hold divergence values
    switcher_array = np.zeros((tHMMobj.num_states, tHMMobj.num_states), dtype="float")

    for state_pred in range(tHMMobj.num_states):
        for state_true in range(tHMMobj.num_states):
            p = [tHMMobj.estimate.E[state_pred].pdf(y) for y in obs_by_state[state_pred]]
            q = [tHMMobj.X[0].E[state_true].pdf(x) for x in obs_by_state[state_pred]]
            switcher_array[state_pred, state_true] = entropy(p, q) + entropy(q, p)

    results_dict["switcher_array"] = switcher_array

    # Create switcher map based on the minimal entropies in the switcher array

    switcher_map = [None] * tHMMobj.num_states

    for row in range(tHMMobj.num_states):
        switcher_row = list(switcher_array[row, :])
        switcher_map[row] = switcher_row.index(min(switcher_row))

    results_dict["switcher_map"] = switcher_map

    # Rearrange the values in the transition matrix
    temp_T = tHMMobj.estimate.T
    for row_idx in range(tHMMobj.num_states):
        for col_idx in range(tHMMobj.num_states):
            temp_T[row_idx, col_idx] = tHMMobj.estimate.T[switcher_map[row_idx], switcher_map[col_idx]]

    results_dict["switched_transition_matrix"] = temp_T
    results_dict["transition_matrix_norm"] = np.linalg.norm(temp_T - tHMMobj.X[0].T)

    # Rearrange the values in the pi vector
    temp_pi = tHMMobj.estimate.pi
    for val_idx in range(tHMMobj.num_states):
        temp_pi[val_idx] = tHMMobj.estimate.pi[switcher_map[val_idx]]

    results_dict["switched_pi_vector"] = temp_pi
    results_dict["pi_vector_norm"] = np.linalg.norm(temp_pi - tHMMobj.X[0].pi)

    # Rearrange the emissions list
    temp_emissions = [None] * tHMMobj.num_states
    for val_idx in range(tHMMobj.num_states):
        temp_emissions[val_idx] = tHMMobj.estimate.E[switcher_map[val_idx]]

    results_dict["switched_emissions"] = temp_emissions

    # Get the estimated parameter values
    results_dict["param_estimates"] = []
    for val_idx in range(tHMMobj.num_states):
        results_dict["param_estimates"].append(temp_emissions[val_idx].params)

    # Get the true parameter values
    results_dict["param_trues"] = []
    for val_idx in range(tHMMobj.num_states):
        results_dict["param_trues"].append(tHMMobj.X[0].E[val_idx].params)

    # 3. Calculate accuracy after switching states
    pred_states_switched = [switcher_map[state] for state in pred_states]
    results_dict["state_counter"] = np.bincount(pred_states_switched)
    results_dict["state_proportions"] = [100 * i / len(pred_states_switched) for i in results_dict["state_counter"]]
    results_dict["state_proportions_0"] = results_dict["state_proportions"][0]
    results_dict["accuracy_before_switching"] = 100 * sum([int(i == j) for i, j in zip(pred_states, true_states)]) / len(true_states)
    results_dict["accuracy_after_switching"] = 100 * sum([int(i == j) for i, j in zip(pred_states_switched, true_states)]) / len(true_states)

    obs_by_state_rand_sampled = []
    for state in range(tHMMobj.num_states):
        full_list = [cell.obs[1] for cell in tHMMobj.X[0].output_lineage if cell.state == state]
        obs_by_state_rand_sampled.append(full_list)

    num2use = min(len(obs_by_state_rand_sampled[0]), len(obs_by_state_rand_sampled[1]))
    if num2use == 0:
        results_dict["wasserstein"] = float("inf")
    else:
        results_dict["wasserstein"] = wasserstein_distance(
            random.sample(obs_by_state_rand_sampled[0], num2use), random.sample(obs_by_state_rand_sampled[1], num2use)
        )

    return results_dict


def run_Results_over(output):
    """
    A function that can be parallelized to speed up figure creation
    This function takes as input:
    output: a list of tuples from the results of running run_Analyze_over
    """
    results_holder = []
    for _, (tHMMobj, pred_states_by_lineage, LL) in enumerate(output):
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
    """
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
        AIC_degrees_of_freedom : the degrees of freedom in AIC calculation (num_states**2 + num_states * number_of_parameters - 1) - same for each lineage
    """
    num_states = tHMMobj.num_states

    number_of_parameters = len(tHMMobj.estimate.E[0].params)
    AIC_degrees_of_freedom = num_states ** 2 + num_states * number_of_parameters - 1

    AIC_value = -2 * LL + 2 * AIC_degrees_of_freedom

    return AIC_value, AIC_degrees_of_freedom


def stateLikelihood(tHMMobj):
    """ We intend to find the likelihood of the states, given observations.
    With Bayes rule we have: P(z | x) = P(x | z) x P(z) / P(x), in which:
    P(x|z) := Emission Likelihood (EL),
    P(z) := Marginal State Distribution (MSD),
    P(x) := Marginal Observation Distribution (NF)
    """
    EL = tHMMobj.get_Emission_Likelihoods()
    MSD = tHMMobj.get_Marginal_State_Distributions()
    NF = get_leaf_Normalizing_Factors(tHMMobj)
    betas = get_leaf_betas(tHMMobj, NF)
    get_nonleaf_NF_and_betas(tHMMobj, NF, betas)
    LL = EL[0] * MSD[0]
    LL[:, 0] = LL[:, 0] / NF[0]
    LL[:, 1] = LL[:, 1] / NF[0]
    return LL


def LLHelperFunc(T, lineageObj):
    """ To calculate the joint probability of state and observations.
    This function, calculates the second term
    P(x_1,...,x_N,z_1,...,z_N) = P(z_1) * prod_{n=2:N}(P(z_n | z_pn)) * prod_{n=1:N}(P(x_n|z_n))
    """
    states = []
    for cell in lineageObj.output_lineage:
        if cell.gen == 1:
            pass
        else:
            states.append(T[cell.parent.state, cell.state])
    return states


def LLFunc(T, pi, tHMMobj, pred_states_by_lineage):
    """ This function calculate the state likelihood, using the joint probability function.
    *** we do the log-transformation to avoid underflow.
    """
    stLikelihood = []
    for indx, lineage in enumerate(tHMMobj.X):
        FirstTerm = pi[lineage.output_lineage[0].state]
        SecondTerm = LLHelperFunc(T, lineage)
        pre_ThirdTerm = tHMMobj.get_Emission_Likelihoods()[indx]
        ThirdTerm = np.zeros(len(lineage.output_lineage))
        for ind, st in enumerate(pred_states_by_lineage[indx]):
            ThirdTerm[ind] = pre_ThirdTerm[ind, st]
        ll = np.log(FirstTerm) + np.sum(np.log(SecondTerm)) + np.sum(np.log(ThirdTerm))
        stLikelihood.append(ll)
    return stLikelihood
