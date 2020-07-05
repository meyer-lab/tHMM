""" Calls the tHMM functions and outputs the parameters needed to generate the Figures. """
from concurrent.futures import ProcessPoolExecutor
import random
import itertools
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import wasserstein_distance

from .tHMM import tHMM


def Analyze(X, num_states, fpi=None, fT=None, fE=None):
    """
    Runs a tHMM and outputs state classification from viterbi, thmm object, normalizing factor, log likelihood, and deltas.

    :param X: A list containing LineageTree objects as lineages.
    :type X: list
    :param num_states: The number of states we want our model to estimate for the given population.
    :type num_states: Int
    :return: The tHMM object
    :rtype: object
    :return: A list containing the lineage-wise predicted states by Viterbi.
    :rtype: list
    :return: Log-likelihood of the normalizing factor for the lineage.
    :rtype: float
    """
    error_holder = []
    for num_tries in range(1, 15):
        try:
            tHMMobj = tHMM(X, num_states=num_states, fpi=fpi, fT=fT, fE=fE)  # build the tHMM class with X
            _, _, _, _, _, LL = tHMMobj.fit()
            break
        except (AssertionError, ZeroDivisionError, RuntimeError) as error:
            error_holder.append(error)
            if num_tries == 14:
                print(
                    f"Caught the following errors: \
                    \n \n {error_holder} \n \n in fitting after multiple {num_tries} runs. \
                    Fitting is breaking after trying {num_tries} times. \
                    If you're facing a ZeroDivisionError or a RuntimeError then the most likely issue \
                    is the estimates of your parameters are returning nonsensible parameters. \
                    Consider changing your parameter estimator. "
                )
                raise

    pred_states_by_lineage = tHMMobj.predict()

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

    :param list_of_populations: A list of populations that contain lineages.
    :type: list
    :param num_states: An integer number of states to identify (a hyper-parameter of our model)
    :type num_states: Int
    """
    list_of_fpi = kwargs.get("list_of_fpi", [None] * len(list_of_populations))
    list_of_fT = kwargs.get("list_of_fT", [None] * len(list_of_populations))
    list_of_fE = kwargs.get("list_of_fE", [None] * len(list_of_populations))
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
    results_dict["total_number_of_cells"] = sum([len(lineage) for lineage in tHMMobj.X])

    true_states_by_lineage = [[cell.state for cell in lineage.output_lineage] for lineage in tHMMobj.X]
    ravel_true_states = np.array([state for sublist in true_states_by_lineage for state in sublist])

    ravel_pred_states = np.array([state for sublist in pred_states_by_lineage for state in sublist])

    # 1. Decide how to switch states based on the state assignment that yields the maximum likelihood
    switcher_map_holder = list(itertools.permutations(list(range(tHMMobj.num_states))))
    new_pred_states_by_lineage_holder = []
    switcher_LL_holder = []
    for _, switcher in enumerate(switcher_map_holder):
        temp_pred_states_by_lineage = []
        for state_assignment in pred_states_by_lineage:
            temp_pred_states_by_lineage.append([switcher[state] for state in state_assignment])
        new_pred_states_by_lineage_holder.append(temp_pred_states_by_lineage)
        switcher_LL_holder.append(np.sum(tHMMobj.log_score(temp_pred_states_by_lineage, pi=tHMMobj.X[0].pi, T=tHMMobj.X[0].T, E=tHMMobj.X[0].E)))
    max_idx = switcher_LL_holder.index(max(switcher_LL_holder))

    # Create switcher map based on the minimal likelihood of different permutations of state
    # assignments
    switcher_map = switcher_map_holder[max_idx]
    switched_pred_states_by_lineage = new_pred_states_by_lineage_holder[max_idx]
    ravel_switched_pred_states = np.array([state for sublist in switched_pred_states_by_lineage for state in sublist])
    results_dict["switcher_map"] = switcher_map
    results_dict["switched_pred_states_by_lineage"] = switched_pred_states_by_lineage
    results_dict["ravel_switched_pred_states"] = ravel_switched_pred_states

    # Rearrange the values in the transition matrix
    temp_T = np.copy(tHMMobj.estimate.T)
    for row in range(tHMMobj.num_states):
        for col in range(tHMMobj.num_states):
            temp_T[row, col] = tHMMobj.estimate.T[switcher_map[row], switcher_map[col]]

    results_dict["switched_transition_matrix"] = temp_T
    results_dict["transition_matrix_norm"] = np.linalg.norm(temp_T - tHMMobj.X[0].T)

    # Rearrange the values in the pi vector
    temp_pi = np.copy(tHMMobj.estimate.pi)
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
        results_dict["param_estimates"].append(results_dict["switched_emissions"][val_idx].params)

    # Get the true parameter values
    results_dict["param_trues"] = []
    for val_idx in range(tHMMobj.num_states):
        results_dict["param_trues"].append(tHMMobj.X[0].E[val_idx].params)

    # 2. Calculate accuracy after switching states
    results_dict["state_counter"] = np.bincount(ravel_switched_pred_states)
    results_dict["state_proportions"] = [100 * i / len(ravel_switched_pred_states) for i in results_dict["state_counter"]]
    results_dict["state_proportions_0"] = results_dict["state_proportions"][0]
    results_dict["accuracy_before_switching"] = 100 * np.mean(ravel_pred_states == ravel_true_states)
    results_dict["accuracy_after_switching"] = 100 * np.mean(ravel_switched_pred_states == ravel_true_states)
    results_dict["balanced_accuracy_score"] = 100 * balanced_accuracy_score(ravel_true_states, ravel_switched_pred_states)

    # 4. Calculate the Wasserstein distance
    obs_index = 1
    if len(tHMMobj.X[0].E[0].params) == 6:
        obs_index = 2

    obs_by_state_rand_sampled = []
    for state in range(tHMMobj.num_states):
        full_list = [cell.obs[obs_index] for cell in tHMMobj.X[0].output_lineage if cell.state == state]
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

    :param output: a list of tuples from the results of running :func:`run_Analyze_over`
    :type output: list
    """
    results_holder = []
    for _, (tHMMobj, pred_states_by_lineage, LL) in enumerate(output):
        results_holder.append(Results(tHMMobj, pred_states_by_lineage, LL))

    return results_holder
