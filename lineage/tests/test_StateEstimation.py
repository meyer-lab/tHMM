""" Unit test file. """
import pytest
import numpy as np

from ..LineageTree import LineageTree
from ..tHMM import tHMM
from ..Analyze import fit_list
from ..states.StateDistributionGamma import (
    atonce_estimator,
    StateDistribution as gamma_state,
)
from ..states.StateDistributionGaPhs import StateDistribution as gamma_statePh


rng = np.random.default_rng(1)


@pytest.mark.parametrize("censored", [0, 3])
def test_estimationEvaluationGamma(censored):
    """
    Evaluates the performance of fitting and the underlying estimator
    by comparing the parameter estimates to their true values.
    """
    pi = np.array([1])
    T = np.array([[1]])
    E_gamma = [gamma_state(bern_p=1.0, gamma_a=7, gamma_scale=4.5)]

    def gen():
        return LineageTree.rand_init(
            pi,
            T,
            E_gamma,
            2**8,
            censor_condition=censored,
            desired_experiment_time=100,
        )

    lineage_gamma = [gen() for _ in range(50)]
    solver_gamma = tHMM(lineage_gamma, 1)  # evaluating for one state
    fit_list([solver_gamma])

    assert solver_gamma.estimate.E[0].dist(E_gamma[0]) < 4.0


def test_atonce_estimator():
    """
    Evaluates the gamma estimator written for fitting all the concentrations at once.
    """
    pi = np.array([1])
    T = np.array([[1]])
    scales1 = [0.2, 0.5, 1.0, 1.5]
    E_gamma = [
        [
            gamma_statePh(
                bern_p1=0.99,
                bern_p2=0.95,
                gamma_a1=70.0,
                gamma_scale1=sc1,
                gamma_a2=140.0,
                gamma_scale2=1.0,
            )
        ]
        for sc1 in scales1
    ]

    def gen(i):
        return LineageTree.rand_init(
            pi, T, E_gamma[i], 2**8, censor_condition=3, desired_experiment_time=250
        )

    lineage_gamma_list = [[gen(i) for _ in range(100)] for i in range(4)]
    solver_gamma_list = [tHMM(lineage_gamma, 1) for lineage_gamma in lineage_gamma_list]

    # check only for one phase
    g1phase_cells = []
    for hmm in solver_gamma_list:
        tmp = np.array(
            [cell.obs for lineage in hmm.X for cell in lineage.output_lineage]
        )
        g1phase_cells.append(tmp[:, np.array([0, 2, 4])])

    # Setup gammas
    gammas_1st = [np.ones(vec.shape[0]) for vec in g1phase_cells]

    atonce_estimator(solver_gamma_list, g1phase_cells, gammas_1st, "G1", 0)
    xout = np.array(
        [solver_gamma_list[0].estimate.E[0].params[2]]
        + [tHMMobj.estimate.E[0].params[3] for tHMMobj in solver_gamma_list]
    )
    assert [
        xout[i + 1] <= xout[i] for i in range(1, 4)
    ]  # check the constraint's condition
    np.testing.assert_allclose(
        xout, np.insert(scales1, 0, 70.0), rtol=0.1
    )  # check for the right solution
