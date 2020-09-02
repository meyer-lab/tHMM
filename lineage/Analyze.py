""" Calls the tHMM functions and outputs the parameters needed to generate the Figures. """
from concurrent.futures import ProcessPoolExecutor
import random
import itertools
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import wasserstein_distance

from .tHMM import tHMM


def Analyze(X, num_states, constant_params=None, fpi=None, fT=None, fE=None):
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
    for num_tries in range(1, 10):
        try:
            tHMMobj = tHMM(X, num_states=num_states, constant_params=constant_params, fpi=fpi, fT=fT, fE=fE)  # build the tHMM class with X
            _, _, _, _, _, LL = tHMMobj.fit()

            tHMMobj2 = tHMM(X, num_states=num_states, constant_params=constant_params, fpi=fpi, fT=fT, fE=fE)  # build the tHMM class with X
            _, _, _, _, _, LL2 = tHMMobj2.fit()

            if LL2 > LL:
                tHMMobj = tHMMobj2
                LL = LL2

            break
        except (AssertionError, ZeroDivisionError, RuntimeError) as error:
            error_holder.append(error)
            if len(error_holder) == 3:
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
    :type num_states: Int or list
    """
    list_of_fpi = kwargs.get("list_of_fpi", [None] * len(list_of_populations))
    list_of_fT = kwargs.get("list_of_fT", [None] * len(list_of_populations))
    list_of_fE = kwargs.get("list_of_fE", [None] * len(list_of_populations))
    const = kwargs.get("constant_params", None)

    if isinstance(num_states, (np.ndarray, list)):
        assert len(num_states) == len(list_of_populations), f"len list population = {len(list_of_populations)}"
    else:
        num_states = np.full(len(list_of_populations), num_states)

    output = []
    if parallel:
        exe = ProcessPoolExecutor()

        prom_holder = []
        for idx, population in enumerate(list_of_populations):
            prom_holder.append(exe.submit(Analyze, population, num_states[idx], constant_params=const, fpi=list_of_fpi[idx], fT=list_of_fT[idx], fE=list_of_fE[idx]))

        output = [prom.result() for prom in prom_holder]
    else:
        for idx, population in enumerate(list_of_populations):
            output.append(Analyze(population, num_states[idx], constant_params=const, fpi=list_of_fpi[idx], fT=list_of_fT[idx], fE=list_of_fE[idx]))

    return output


def Results(tHMMobj, pred_states_by_lineage, LL):
    """
    This function calculates several results of fitting a synthetic lineage.
    """
    # Instantiating a dictionary to hold the various metrics of accuracy and scoring for the results of our method
    results_dict = {}
    results_dict["total_number_of_lineages"] = len(tHMMobj.X)
    results_dict["LL"] = LL
    results_dict["total_number_of_cells"] = sum([len(lineage.output_lineage) for lineage in tHMMobj.X])

    true_states_by_lineage = [[cell.state for cell in lineage.output_lineage] for lineage in tHMMobj.X]
    ravel_true_states = np.array([state for sublist in true_states_by_lineage for state in sublist])

    # 1. Decide how to switch states based on the state assignment that yields the maximum likelihood
    switcher_map_holder = list(itertools.permutations(list(range(tHMMobj.num_states))))
    switcher_LL_holder = np.empty(len(switcher_map_holder))

    pi_arg = tHMMobj.X[0].pi
    T_arg = tHMMobj.X[0].T
    E_arg = tHMMobj.X[0].E
    if tHMMobj.fpi is not None:
        pi_arg = tHMMobj.fpi
    if tHMMobj.fT is not None:
        T_arg = tHMMobj.fT
    if tHMMobj.fE is not None:
        E_arg = tHMMobj.fE

    for ii, switcher in enumerate(switcher_map_holder):
        sw_states = [[switcher[st] for st in st_ass] for st_ass in pred_states_by_lineage]
        switcher_LL_holder[ii] = np.sum(tHMMobj.log_score(sw_states, pi=pi_arg, T=T_arg, E=E_arg))

    # Create switcher map based on the max likelihood of different permutations of state assignments
    switcher_map = np.array(switcher_map_holder[np.argmax(switcher_LL_holder)])
    results_dict["switcher_map"] = switcher_map
    ravel_switched_pred_states = np.array([switcher_map[st] for sublist in pred_states_by_lineage for st in sublist])

    # Rearrange the values in the transition matrix
    results_dict["switched_transition_matrix"] = tHMMobj.estimate.T[switcher_map, switcher_map]
    results_dict["transition_matrix_norm"] = np.linalg.norm(results_dict["switched_transition_matrix"] - tHMMobj.X[0].T)

    # Rearrange the values in the pi vector
    results_dict["switched_pi_vector"] = tHMMobj.estimate.pi[switcher_map]
    results_dict["pi_vector_norm"] = np.linalg.norm(results_dict["switched_pi_vector"] - tHMMobj.X[0].pi)

    # Rearrange the emissions list
    results_dict["switched_emissions"] = [tHMMobj.estimate.E[switcher_map[x]] for x in range(tHMMobj.num_states)]

    # Get the estimated parameter values
    results_dict["param_estimates"] = [results_dict["switched_emissions"][x].params for x in range(tHMMobj.num_states)]

    # Get the true parameter values
    results_dict["param_trues"] = [tHMMobj.X[0].E[x].params for x in range(tHMMobj.num_states)]

    # 2. Calculate accuracy after switching states
    results_dict["state_counter"] = np.bincount(ravel_switched_pred_states)
    results_dict["state_proportions"] = [100 * i / len(ravel_switched_pred_states) for i in results_dict["state_counter"]]
    results_dict["state_proportions_0"] = results_dict["state_proportions"][0]
    results_dict["accuracy_after_switching"] = 100 * np.mean(ravel_true_states == ravel_switched_pred_states)
    results_dict["balanced_accuracy_score"] = 100 * balanced_accuracy_score(ravel_true_states, ravel_switched_pred_states)

    # 4. Calculate the Wasserstein distance
    results_dict["wasserstein"] = tHMMobj.X[0].E[0].dist(tHMMobj.X[0].E[1])

    return results_dict


def run_Results_over(output, parallel=True):
    """
    A function that can be parallelized to speed up figure creation

    :param output: a list of tuples from the results of running :func:`run_Analyze_over`
    :type output: list
    """
    if parallel:
        exe = ProcessPoolExecutor()
        prom_holder = [exe.submit(Results, *x) for x in output]
        results_holder = [prom.result() for prom in prom_holder]
    else:
        results_holder = [Results(*x) for x in output]

    return results_holder
