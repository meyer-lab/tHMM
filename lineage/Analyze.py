""" Calls the tHMM functions and outputs the parameters needed to generate the Figures. """
import itertools
import numpy as np
from concurrent.futures import ProcessPoolExecutor, Future, Executor
from sklearn.metrics import rand_score, confusion_matrix
from .tHMM import tHMM, fit_list
from typing import Any, Tuple, Union


class DummyExecutor(Executor):
    def submit(self, fn, *args, **kwargs):
        f = Future()
        result = fn(*args, **kwargs)
        f.set_result(result)
        return f


def Analyze(X: list, num_states: int, **kwargs) -> Tuple[object, float]:
    """ Runs the model and outputs the tHMM object, state assignments, and likelihood.
    :param X: The list of LineageTree populations.
    :param num_states: The number of states that we want to run the model for.
    :return tHMMobj_list: The tHMMobj after fitting corresponding to the given LineageTree population.
    :return st: The nested list of states assigned to cells, with the order of cells from root to leaf in each lineage, and generation.
    :return LL: The log-likelihood of the fitted model.
    """
    tHMMobj_list, LL, _ = Analyze_list([X], num_states, **kwargs)
    return tHMMobj_list[0], LL


def Analyze_list(Population_list: list, num_states: int, **kwargs) -> Tuple[list, float]:
    """ This function runs the analyze function for the case when we want to fit multiple conditions at the same time.
    :param Population_list: The list of cell populations to run the analyze function on.
    :param num_states: The number of states that we want to run the model for.
    :return tHMMobj_list: The tHMMobj after fitting corresponding to the given LineageTree population.
    :return pred_states_by_lineage_by_conc: The list of cells in each lineage with states assigned to each cell.
    :return LL: The log-likelihood of the fitted model.
    """

    tHMMobj_list = [tHMM(X, num_states=num_states, **kwargs) for X in Population_list]  # build the tHMM class with X
    _, _, _, gammas, LL = fit_list(tHMMobj_list)

    for _ in range(5):
        tHMMobj_list2 = [tHMM(X, num_states=num_states, **kwargs) for X in Population_list]  # build the tHMM class with X
        _, _, _, gammas2, LL2 = fit_list(tHMMobj_list2)

        if LL2 > LL:
            tHMMobj_list = tHMMobj_list2
            LL = LL2
            gammas = gammas2

    return tHMMobj_list, LL, gammas


def run_Analyze_over(list_of_populations: list, num_states: np.ndarray, parallel=True, atonce=False, **kwargs) -> list:
    """
    A function that can be parallelized to speed up figure creation.

    This function is the outermost for-loop we will end up using
    when analyzing heterogenous populations or lineages.

    Analyze is the bottleneck in the figure creation process. The
    rest of the code involved in figure creation deals with collecting
    and computing certain statistics, most of which can be done in an
    additional for loop over the results from Analyze.
    :param list_of_populations: The list of cell populations to run the analyze function on.
    :param num_states: The number of states that we want to run the model for.
    :return output: The list of results from fitting a lineage.
    """
    list_of_fpi = kwargs.get("list_of_fpi", [None] * len(list_of_populations))
    list_of_fT = kwargs.get("list_of_fT", [None] * len(list_of_populations))
    list_of_fE = kwargs.get("list_of_fE", [None] * len(list_of_populations))

    if isinstance(num_states, (np.ndarray, list)):
        assert len(num_states) == len(list_of_populations)
    else:
        num_states = np.full(len(list_of_populations), num_states)

    output = []
    exe: Union[ProcessPoolExecutor, DummyExecutor]
    if parallel:
        exe = ProcessPoolExecutor()
    else:
        exe = DummyExecutor()

    prom_holder = []
    for idx, population in enumerate(list_of_populations):
        if atonce:  # if we are running all the concentration simultaneously, they should be given to Analyze_list() specifically in the case of figure 9
            prom_holder.append(exe.submit(Analyze_list, population, num_states[idx], fpi=list_of_fpi[idx], fT=list_of_fT[idx], fE=list_of_fE[idx]))
        else:  # if we are not fitting all conditions at once, we need to pass the populations to the Analyze()
            prom_holder.append(exe.submit(Analyze, population, num_states[idx], fpi=list_of_fpi[idx], fT=list_of_fT[idx], fE=list_of_fE[idx]))

    output = [prom.result() for prom in prom_holder]

    return output


def Results(tHMMobj, LL: float) -> dict[str, Any]:
    """
    This function calculates several results of fitting a synthetic lineage and stores it in a dictionary.
    The dictionary contains the total number of lineages, the log likelihood of state assignments, and
    the total number of cells. It also contains metrics such as the accuracy of state assignment predictions,
    the distance between two distributions, and the Wasserstein distance between two states.
    :param tHMMobj: An instantiation of the tHMM class.
    :param LL: The log-likelihood of the fitted model.
    :return results_dict: A dictionary containing metrics of accuracy and scoring for the results of fitting a lineage.
    """
    # Instantiating a dictionary to hold the various metrics of accuracy and scoring for the results of our method
    results_dict: dict[str, Any]
    results_dict = {}
    # To find the switcher map for states based on log-likelihood
    switcher_map = cheat(tHMMobj)
    tHMMobj, pred_states = permute_states(tHMMobj, switcher_map)

    results_dict["total_number_of_lineages"] = len(tHMMobj.X)
    results_dict["LL"] = LL
    results_dict["total_number_of_cells"] = sum([len(lineage.output_lineage) for lineage in tHMMobj.X])

    true_states_by_lineage = [[cell.state for cell in lineage.output_lineage] for lineage in tHMMobj.X]

    results_dict["transition_matrix_similarity"] = np.linalg.norm(tHMMobj.estimate.T - tHMMobj.X[0].T)

    results_dict["pi_similarity"] = np.linalg.norm(tHMMobj.X[0].pi - tHMMobj.estimate.pi)

    # Get the estimated parameter values
    results_dict["param_estimates"] = [tHMMobj.estimate.E[x].params for x in range(tHMMobj.num_states)]

    # Get the true parameter values
    results_dict["param_trues"] = [tHMMobj.X[0].E[x].params for x in range(tHMMobj.num_states)]

    # Get the distance between distributions of two states
    results_dict["distribution distance 0"] = tHMMobj.estimate.E[0].dist(tHMMobj.X[0].E[0])
    results_dict["distribution distance 1"] = tHMMobj.estimate.E[1].dist(tHMMobj.X[0].E[1])

    # 2. Calculate accuracy after switching states
    results_dict["state_counter"] = np.bincount(pred_states[0])
    results_dict["state_proportions"] = [100.0 * i / len(pred_states[0]) for i in results_dict["state_counter"]]
    results_dict["state_proportions_0"] = results_dict["state_proportions"][0]
    results_dict["state_similarity"] = 100.0 * rand_score(list(itertools.chain(*true_states_by_lineage)), list(itertools.chain(*pred_states)))
    results_dict["confusion_matrix"] = confusion_matrix(list(itertools.chain(*true_states_by_lineage)), list(itertools.chain(*pred_states)))

    # 4. Calculate the Wasserstein distance
    results_dict["wasserstein"] = tHMMobj.X[0].E[0].dist(tHMMobj.X[0].E[1])

    return results_dict


def run_Results_over(output: list, parallel=True) -> list:
    """
    A function that can be parallelized to speed up figure creation.
    Output is a list of tuples from the results of running :func:`run_Analyze_over`
    :param output: The list of results from fitting a lineage.
    :param parallel: True if we have multiple conditions to run at once, False if no parallel fitting.
    """
    exe: Union[ProcessPoolExecutor, DummyExecutor]
    if parallel:
        exe = ProcessPoolExecutor()
    else:
        exe = DummyExecutor()

    prom_holder = [exe.submit(Results, *x) for x in output]
    return [prom.result() for prom in prom_holder]


def permute_states(tHMMobj: Any, switch_map: np.ndarray) -> Tuple[Any, list]:
    """
    This function takes the tHMMobj and the predicted states,
    and finds out whether we need to switch the state identities or not based on the likelihood.
    :param tHMMobj: An instantiation of the tHMM class.
    :param switch_map: An array of the likelihood of predicted states.
    :return tHMMobj: An instantiation of the tHMM class.
    :return pred_states_switched: A list of lineages with switched states.
    """
    pred_states = tHMMobj.predict()

    pred_states_switched = [np.array([switch_map[st] for sublist in pred_states for st in sublist])]

    # Rearrange the values in the transition matrix
    tHMMobj.estimate.T = tHMMobj.estimate.T[switch_map, :]
    tHMMobj.estimate.T = tHMMobj.estimate.T[:, switch_map]

    # Rearrange the values in the pi vector
    tHMMobj.estimate.pi = tHMMobj.estimate.pi[switch_map]

    # Rearrange the emissions list
    tHMMobj.estimate.E = [tHMMobj.estimate.E[ii] for ii in switch_map]

    return tHMMobj, pred_states_switched


def cheat(tHMMobj):
    """
    Find out the map between the assigned and true states by finding the closest pairs of parameters.
    Works for synthetic data that we know the true parameters.
    """

    true_params = np.array([tHMMobj.X[0].E[i].params for i in range(tHMMobj.num_states)])
    est_params = np.array([tHMMobj.estimate.E[i].params for i in range(tHMMobj.num_states)])
    assert(est_params.shape == true_params.shape)

    mapp = []
    for i, ps in enumerate(true_params):

        dist = []  # find the norm2 distance between pairs of true and estimated parameters
        for est_p in est_params:
            dist.append(np.linalg.norm(ps - est_p))

        mapp.append(np.argmin(dist))
        est_params[np.argmin(dist), :] = -1000.0

    return mapp
