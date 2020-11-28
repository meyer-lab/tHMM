""" Unit test file. """
import pytest
import numpy as np
import scipy.stats as sp

from ..BaumWelch import do_E_step
from ..LineageTree import LineageTree
from ..tHMM import tHMM
from ..states.stateCommon import gamma_estimator_atonce 
from ..states.StateDistributionGamma import atonce_estimator, StateDistribution as gamma_state


@pytest.mark.parametrize("censored", [0, 3])
@pytest.mark.parametrize("constant_shape", [True, False])
def test_estimationEvaluationGamma(censored, constant_shape):
    """
    Evaluates the performance of fitting and the underlying estimator
    by comparing the parameter estimates to their true values.
    """
    pi = np.array([1])
    T = np.array([[1]])
    E_gamma = [gamma_state(bern_p=1., gamma_a=7, gamma_scale=4.5)]

    if constant_shape:
        E_gamma[0].const_shape = E_gamma[0].params[1]

    def gen(): return LineageTree.init_from_parameters(pi, T, E_gamma, 2**8, censor_condition=censored, desired_experiment_time=100)
    lineage_gamma = [gen() for _ in range(50)]
    solver_gamma = tHMM(lineage_gamma, 1)  # evaluating for one state
    solver_gamma.fit()

    assert solver_gamma.estimate.E[0].dist(E_gamma[0]) < 4.0

def test_atonce_estimator():
    """ 
    Evaluates the gamma estimator written for fitting all the concentrations at once.
    """
    pi = np.array([1])
    T = np.array([[1]])
    scales = [4.0, 5.0, 6.0, 7.0]
    E_gamma = [[gamma_state(bern_p=1., gamma_a=7., gamma_scale=sc)] for sc in scales]

    def gen(i): return LineageTree.init_from_parameters(pi, T, E_gamma[i], 2**8, censor_condition=3, desired_experiment_time=250)
    lineage_gamma_list = [[gen(i) for _ in range(50)] for i in range(4)]
    solver_gamma_list = [tHMM(lineage_gamma, 1) for lineage_gamma in lineage_gamma_list]
    list_gammas = [do_E_step(tHMMoj)[3] for tHMMoj in solver_gamma_list]
    x_list = []
    for hmm in solver_gamma_list:
        tmp = []
        for lineage in hmm.X:
            for cell in lineage.output_lineage:
                tmp.append(cell.obs)
        x_list.append(tmp)
    xout = atonce_estimator(x_list, list_gammas)
    assert np.all(np.abs(xout - ([7.] + scales)) <= 1.5)
