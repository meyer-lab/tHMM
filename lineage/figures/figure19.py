""" To plot a summary of cross validation. """
import numpy as np
import itertools as it
from ..LineageTree import LineageTree, hide_observation
from ..Analyze import cv_likelihood, Analyze_list, Results
from ..tHMM import tHMM, fit_list
from ..BaumWelch import calculate_stationary
from ..states.StateDistributionGamma import StateDistribution

T = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=float)

# pi: the initial probability vector
pi = calculate_stationary(T)

# bern, gamma_a, gamma_scale
state0 = StateDistribution(0.99, 100, 0.1)
state1 = StateDistribution(0.75, 80, 0.5)
E = [state0, state1]

def cv():
    complete_lineage = LineageTree.init_from_parameters(pi, T, E, 31)
    true_states_by_lineage = [cell.state for cell in complete_lineage.output_lineage]

    lineage = hide_observation(complete_lineage)
    tHMMobj, LL = Analyze_list([[lineage]], 2)
    results = Results(tHMMobj[0], LL)

    print("confusion_mat", results["confusion_matrix"])
    print("accuracy", results["state_similarity"])

    pred, true = cv_likelihood(tHMMobj[0])
    print("pred ", pred)
    print("true", true)