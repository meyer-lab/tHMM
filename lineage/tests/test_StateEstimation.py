""" Unit test file. """
import pytest
import numpy as np

from ..LineageTree import LineageTree
from ..tHMM import tHMM
from ..states.StateDistributionGamma import StateDistribution as gamma_state


@pytest.mark.parametrize("censored", [True, False])
def test_estimationEvaluationGamma(censored):
    """
    Evaluates the performance of fitting and the underlying estimator
    by comparing the parameter estimates to their true values.
    """
    pi = np.array([1])
    T = np.array([[1]])
    E_gamma = [gamma_state(bern_p=1., gamma_a=7, gamma_scale=4.5)]

    if censored:
        def gen(): return LineageTree.init_from_parameters(pi, T, E_gamma, 2**9, censor_condition=3, desired_experiment_time=100)
    else:
        def gen(): return LineageTree.init_from_parameters(pi, T, E_gamma, 2**9)

    lineage_gamma = [gen() for _ in range(20)]
    solver_gamma = tHMM(lineage_gamma, 1)  # evaluating for one state
    solver_gamma.fit()

    assert solver_gamma.estimate.E[0].dist(E_gamma[0]) < 3.0
