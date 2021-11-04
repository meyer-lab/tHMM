""" Calls the tHMM functions and outputs the parameters needed to generate the Figures. """
import itertools
import numpy as np
from concurrent.futures import ProcessPoolExecutor, Future, Executor
from sklearn.metrics import rand_score
from .tHMM import tHMM, fit_list
from typing import Any, Tuple, Union


class DummyExecutor(Executor):
    def submit(self, fn, *args, **kwargs):
        f = Future()
        result = fn(*args, **kwargs)
        f.set_result(result)
        return f


def Analyze(X: list, num_states: int, **kwargs) -> Tuple[object, int, float]:
    """ Runs a tHMM and outputs the tHMM object, state assignments, and likelihood. """
    tHMMobj_list, st, LL = Analyze_list([X], num_states, **kwargs)
    return tHMMobj_list[0], st[0], LL


def Analyze_list(Population_list: list, num_states: int, **kwargs) -> Tuple[list, list, float]:
    """ This function runs the analyze for the case when we want to fit the experimental data. (fig 11)"""

    tHMMobj_list = [tHMM(X, num_states=num_states, **kwargs) for X in Population_list]  # build the tHMM class with X
    _, _, _, _, LL = fit_list(tHMMobj_list)

    for _ in range(2):
        tHMMobj_list2 = [tHMM(X, num_states=num_states, **kwargs) for X in Population_list]  # build the tHMM class with X
        _, _, _, _, LL2 = fit_list(tHMMobj_list2)

        if LL2 > LL:
            tHMMobj_list = tHMMobj_list2
            LL = LL2

    pred_states_by_lineage_by_conc = [tHMMobj.predict() for tHMMobj in tHMMobj_list]

    return tHMMobj_list, pred_states_by_lineage_by_conc, LL


def run_Analyze_over(list_of_populations: list, num_states: np.ndarray, parallel=True, atonce=False, **kwargs) -> list:
    """
    A function that can be parallelized to speed up figure creation.

    This function is the outermost for-loop we will end up using
    when analyzing heterogenous populations or lineages.

    Analyze is the bottleneck in the figure creation process. The
    rest of the code involved in figure creation deals with collecting
    and computing certain statistics, most of which can be done in an
    additional for loop over the results from Analyze.
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


def Results(tHMMobj, pred_states_by_lineage: list, LL: float) -> dict[str, Any]:
    """
    This function calculates several results of fitting a synthetic lineage and stores it in a dictionary.
    The dictionary contains the total number of lineages, the log likelihood of state assignments, and
    the total number of cells. It also contains metrics such as the accuracy of state assignment predictions,
    the distance between two distributions, and the Wasserstein distance between two states.

    """
    # Instantiating a dictionary to hold the various metrics of accuracy and scoring for the results of our method
    results_dict: dict[str, Any]
    results_dict = {}
    results_dict["total_number_of_lineages"] = len(tHMMobj.X)
    results_dict["LL"] = LL
    results_dict["total_number_of_cells"] = sum([len(lineage.output_lineage) for lineage in tHMMobj.X])

    true_states_by_lineage = [[cell.state for cell in lineage.output_lineage] for lineage in tHMMobj.X]

    results_dict["transition_matrix_similarity"] = np.linalg.norm(tHMMobj.estimate.T - tHMMobj.X[0].T)

    results_dict["pi_similarity"] = rand_score(tHMMobj.estimate.pi, tHMMobj.X[0].pi)

    # Get the estimated parameter values
    results_dict["param_estimates"] = [tHMMobj.estimate.E[x].params for x in range(tHMMobj.num_states)]

    # Get the true parameter values
    results_dict["param_trues"] = [tHMMobj.X[0].E[x].params for x in range(tHMMobj.num_states)]

    # Get the distance between distributions of two states
    results_dict["distribution distance 0"] = tHMMobj.estimate.E[0].dist(tHMMobj.X[0].E[0])
    results_dict["distribution distance 1"] = tHMMobj.estimate.E[1].dist(tHMMobj.X[0].E[1])

    # 2. Calculate accuracy after switching states
    results_dict["state_counter"] = np.bincount(pred_states_by_lineage[0])
    results_dict["state_proportions"] = [100 * i / len(pred_states_by_lineage) for i in results_dict["state_counter"]]
    results_dict["state_proportions_0"] = results_dict["state_proportions"][0]
    results_dict["state_similarity"] = 100 * rand_score(list(itertools.chain(*true_states_by_lineage)), list(itertools.chain(*pred_states_by_lineage)))

    # 4. Calculate the Wasserstein distance
    results_dict["wasserstein"] = tHMMobj.X[0].E[0].dist(tHMMobj.X[0].E[1])

    return results_dict


def run_Results_over(output: list, parallel=True) -> list:
    """
    A function that can be parallelized to speed up figure creation.
    Output is a list of tuples from the results of running :func:`run_Analyze_over`
    """
    exe: Union[ProcessPoolExecutor, DummyExecutor]
    if parallel:
        exe = ProcessPoolExecutor()
    else:
        exe = DummyExecutor()

    prom_holder = [exe.submit(Results, *x) for x in output]
    return [prom.result() for prom in prom_holder]
