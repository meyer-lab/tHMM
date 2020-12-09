""" Unit test file. """
import pytest
import numpy as np
import scipy.stats as sp

from ..BaumWelch import do_E_step
from ..LineageTree import LineageTree
from ..tHMM import tHMM
from ..states.stateCommon import gamma_estimator_atonce
from ..states.StateDistributionGamma import atonce_estimator, StateDistribution as gamma_state
from ..states.StateDistributionGaPhs import StateDistribution as gamma_statePh


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
    scales1 = [4., 5., 6., 7.]
    E_gamma = [[gamma_statePh(bern_p1=0.99, bern_p2=0.95, gamma_a1=7.0, gamma_scale1=sc1, gamma_a2=14.0, gamma_scale2=1.)] for sc1 in scales1]

    def gen(i): return LineageTree.init_from_parameters(pi, T, E_gamma[i], 2**8, censor_condition=3, desired_experiment_time=250)
    lineage_gamma_list = [[gen(i) for _ in range(50)] for i in range(4)]
    solver_gamma_list = [tHMM(lineage_gamma, 1) for lineage_gamma in lineage_gamma_list]
    list_gammas = [do_E_step(tHMMoj)[3] for tHMMoj in solver_gamma_list]

    gms = []
    for gm in list_gammas:
        gms.append(np.vstack(gm))
    # reshape the gammas so that the list contains for the only state
    gammas_1st = [array[:, 0] for array in gms]

    # check only for one phase
    g1phase_cells = []
    for hmm in solver_gamma_list:
        tmp = np.array([cell.obs for lineage in hmm.X for cell in lineage.output_lineage])
        g1phase_cells.append(tmp[:, np.array([0, 2, 4])])

    xout, _ = atonce_estimator(g1phase_cells, gammas_1st)
    assert [xout[i + 1] <= xout[i] for i in range(1, 4)]  # check the constraint's condition
    assert np.all(np.abs(xout - ([7.] + scales1)) <= 1.5)  # check optimization is good
