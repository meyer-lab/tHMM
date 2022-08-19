""" Unit test file. """
from copy import deepcopy
import pytest
import numpy as np
import pickle
from sklearn.metrics import rand_score
from ..states.StateDistributionGaPhs import StateDistribution as phaseStateDist
from ..BaumWelch import do_E_step, do_M_E_step, calculate_log_likelihood, calculate_stationary
from ..LineageTree import LineageTree
from ..tHMM import tHMM
from ..Analyze import fit_list
from ..figures.common import pi, T, E


@pytest.mark.parametrize("cens", [0, 2])
@pytest.mark.parametrize("nStates", [1, 2, 3])
def test_BW(cens, nStates):
    """ This tests that one step of Baum-Welch increases the likelihood of the fit. """
    X = LineageTree.init_from_parameters(pi, T, E, desired_num_cells=(2 ** 7) - 1, desired_experimental_time=200, censor_condition=cens)
    tHMMobj = tHMM([X], num_states=nStates)  # build the tHMM class with X

    # Test cases below
    # Get the likelihoods before fitting
    _, NF, _, _ = do_E_step(tHMMobj)
    LL_before = calculate_log_likelihood(NF)
    assert np.isfinite(LL_before)

    # Get the likelihoods after fitting
    _, NF_after, _, _, new_LL_list_after = fit_list([tHMMobj], max_iter=3)

    LL_after = calculate_log_likelihood(NF_after[0])
    assert np.isfinite(LL_after)
    assert np.isfinite(new_LL_list_after)
    assert LL_after > LL_before


def test_fit_seed():
    """ Test that we can set the seed to provide reproducible results. """
    X = LineageTree.init_from_parameters(pi, T, E, desired_num_cells=(2 ** 7) - 1, desired_experimental_time=200)
    tHMMobj = tHMM([X], num_states=2)  # build the tHMM class with X

    # Get the likelihoods after fitting
    _, NFone, _, _, LLone = fit_list([deepcopy(tHMMobj)], max_iter=3, rng=1)
    _, NFtwo, _, _, LLtwo = fit_list([deepcopy(tHMMobj)], max_iter=3, rng=1)
    assert LLone == LLtwo
    np.testing.assert_allclose(NFone, NFtwo)

    _, _, _, _, LLone = fit_list([deepcopy(tHMMobj)], max_iter=3)
    _, _, _, _, LLtwo = fit_list([deepcopy(tHMMobj)], max_iter=3)
    assert LLone != LLtwo


pik1 = open("gemcitabines.pkl", "rb")
gmc = []
for i in range(4):
    gmc.append(pickle.load(pik1))

# model parameters for lapatinib 25nM
E3 = gmc[1].estimate.E
T3 = gmc[1].estimate.T
pi3 = gmc[1].estimate.pi


@pytest.mark.parametrize("cens", [0, 3])
def test_E_step(cens):
    """ This tests that given the true model parameters, can it estimate the states correctly."""
    T = np.array([[0.6, 0.1, 0.1, 0.1, 0.1], [0.05, 0.8, 0.05, 0.05, 0.05], [0.01, 0.1, 0.7, 0.09, 0.1], [0.1, 0.1, 0.05, 0.7, 0.05], [0.1, 0.1, 0.05, 0.05, 0.7]], dtype=float)

    # pi: the initial probability vector
    pi = calculate_stationary(T)

    state0 = phaseStateDist(0.99, 0.95, 50, 0.2, 100, 0.1)
    state1 = phaseStateDist(0.95, 0.9, 75, 0.2, 150, 0.1)
    state2 = phaseStateDist(0.9, 0.85, 100, 0.2, 200, 0.1)
    state3 = phaseStateDist(0.92, 0.95, 150, 0.2, 250, 0.1)
    state4 = phaseStateDist(0.99, 0.85, 200, 0.2, 300, 0.1)
    E = [state0, state1, state2, state3, state4]
    population = []
    for _ in range(200):
        # make sure we have enough cells in the lineage.
        X = LineageTree.init_from_parameters(pi, T, E, desired_num_cells=(2 ** 6) - 1, desired_experimental_time=150, censor_condition=cens)
        while len(X.output_lineage) < 5:
            X = LineageTree.init_from_parameters(pi, T, E, desired_num_cells=(2 ** 6) - 1, desired_experimental_time=150, censor_condition=cens)
        population.append(X)

    tHMMobj = tHMM(population, num_states=5)  # build the tHMM class with X
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
    for _ in range(500):
        # make sure we have enough cells in the lineage.
        X = LineageTree.init_from_parameters(pi3, T3, E3, desired_num_cells=(2 ** 5) - 1, desired_experimental_time=100, censor_condition=cens)
        while len(X.output_lineage) < 4:
            X = LineageTree.init_from_parameters(pi3, T3, E3, desired_num_cells=(2 ** 5) - 1, desired_experimental_time=100, censor_condition=cens)
        population.append(X)

    tHMMobj = tHMM(population, num_states=gmc[1].num_states)
    gammas = [np.zeros((len(lineage.output_lineage), tHMMobj.num_states)) for lineage in tHMMobj.X]

    # create the gamma matrix (N x K) that shows the probability of a cell n being in state k from the true state assignments.
    for idx, g_lin in enumerate(gammas):
        for i in range(g_lin.shape[0]):
            g_lin[i, tHMMobj.X[idx].output_lineage[i].state] = 1

    do_M_E_step(tHMMobj, gammas)
    # Test that parameter values match our input
    for i in range(gmc[1].num_states):
        np.testing.assert_allclose(tHMMobj.estimate.E[i].params, E3[i].params, rtol=0.1)
