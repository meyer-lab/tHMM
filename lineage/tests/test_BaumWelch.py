""" Unit test file. """
from copy import deepcopy
import pytest
import numpy as np
from sklearn.metrics import rand_score
from ..states.StateDistributionGaPhs import StateDistribution as phaseStateDist
from ..BaumWelch import do_E_step, do_M_E_step, calculate_log_likelihood, calculate_stationary
from ..LineageTree import LineageTree
from ..tHMM import tHMM
from ..Analyze import fit_list


T = np.array([[0.6, 0.1, 0.1, 0.1, 0.1], [0.05, 0.8, 0.05, 0.05, 0.05], [0.01, 0.1, 0.7, 0.09, 0.1], [0.1, 0.1, 0.05, 0.7, 0.05], [0.1, 0.1, 0.05, 0.05, 0.7]], dtype=float)

# pi: the initial probability vector
pi = calculate_stationary(T)

state0 = phaseStateDist(0.99, 0.95, 60, 0.2, 100, 0.1)
state1 = phaseStateDist(0.95, 0.9, 75, 0.2, 150, 0.1)
state2 = phaseStateDist(0.9, 0.85, 100, 0.2, 200, 0.1)
state3 = phaseStateDist(0.92, 0.95, 150, 0.2, 250, 0.1)
state4 = phaseStateDist(0.99, 0.85, 200, 0.2, 300, 0.1)
E = [state0, state1, state2, state3, state4]

expt_time = 300
num_cells = 200

rng = np.random.default_rng(4)


@pytest.mark.parametrize("cens", [0, 2])
@pytest.mark.parametrize("nStates", [1, 2, 3])
def test_BW(cens, nStates):
    """ This tests that one step of Baum-Welch increases the likelihood of the fit. """
    X = LineageTree.rand_init(pi, T, E, desired_num_cells=num_cells, desired_experiment_time=expt_time, censor_condition=cens)
    tHMMobj = tHMM([X], num_states=nStates)  # build the tHMM class with X

    # Test cases below
    # Get the likelihoods before fitting
    _, NF, _, _ = do_E_step(tHMMobj)
    LL_before = calculate_log_likelihood(NF)
    assert np.isfinite(LL_before)

    # Get the likelihoods after fitting
    NF_after, _, new_LL_list_after = fit_list([tHMMobj], max_iter=3)

    LL_after = calculate_log_likelihood(NF_after[0])
    assert np.isfinite(LL_after)
    assert np.isfinite(new_LL_list_after)
    assert LL_after > LL_before


def test_fit_seed():
    """ Test that we can set the seed to provide reproducible results. """
    X = LineageTree.rand_init(pi, T, E, desired_num_cells=num_cells, desired_experiment_time=expt_time)
    tHMMobj = tHMM([X], num_states=2)  # build the tHMM class with X

    # Get the likelihoods after fitting
    NFone, _, LLone = fit_list([deepcopy(tHMMobj)], max_iter=3, rng=1)
    NFtwo, _, LLtwo = fit_list([deepcopy(tHMMobj)], max_iter=3, rng=1)
    assert LLone == LLtwo
    np.testing.assert_allclose(NFone, NFtwo)

    _, _, LLone = fit_list([deepcopy(tHMMobj)], max_iter=3)
    _, _, LLtwo = fit_list([deepcopy(tHMMobj)], max_iter=3)
    assert LLone != LLtwo


@pytest.mark.parametrize("cens", [0, 3])
def test_E_step(cens):
    """ This tests that given the true model parameters, can it estimate the states correctly."""

    population = []
    for _ in range(30):
        # make sure we have enough cells in the lineage.
        X = LineageTree.rand_init(pi, T, E, desired_num_cells=num_cells, desired_experiment_time=expt_time, censor_condition=cens, rng=rng)
        population.append(X)

    tHMMobj = tHMM(population, num_states=5, rng=rng)  # build the tHMM class with X
    tHMMobj.estimate.pi = pi
    tHMMobj.estimate.T = T
    tHMMobj.estimate.E = E

    do_E_step(tHMMobj)
    pred_states = tHMMobj.predict()
    true_states = [cell.state for cell in tHMMobj.X[0].output_lineage]

    assert rand_score(true_states, pred_states[0]) >= 0.9


@pytest.mark.parametrize("cens", [0, 3])
def test_M_step(cens):
    """ The M step of the BW. check the emission parameters if the true states are given. """
    population = []
    for _ in range(30):
        # make sure we have enough cells in the lineage.
        X = LineageTree.rand_init(pi, T, E, desired_num_cells=num_cells, desired_experiment_time=expt_time, censor_condition=cens, rng=rng)
        population.append(X)

    tHMMobj = tHMM(population, num_states=len(E), rng=rng)
    gammas = [np.zeros((len(lineage.output_lineage), tHMMobj.num_states)) for lineage in tHMMobj.X]

    # create the gamma matrix (N x K) that shows the probability of a cell n being in state k from the true state assignments.
    for idx, g_lin in enumerate(gammas):
        for i in range(g_lin.shape[0]):
            g_lin[i, tHMMobj.X[idx].output_lineage[i].state] = 1

    do_M_E_step(tHMMobj, gammas)
    # Test that parameter values match our input
    for i in range(len(E)):
        np.testing.assert_allclose(tHMMobj.estimate.E[i].params, E[i].params, rtol=0.2)
