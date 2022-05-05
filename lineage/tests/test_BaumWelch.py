""" Unit test file. """
import pytest
import numpy as np
from sklearn.metrics import rand_score
from ..BaumWelch import do_E_step, do_M_E_step, calculate_log_likelihood
from ..LineageTree import LineageTree
from ..tHMM import tHMM
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
    _, _, NF_after, _, _, new_LL_list_after = tHMMobj.fit(max_iter=3)
    LL_after = calculate_log_likelihood(NF_after)
    assert np.isfinite(LL_after)
    assert np.isfinite(new_LL_list_after)
    assert LL_after > LL_before

@pytest.mark.parametrize("cens", [0, 2])
def test_E_step(cens):
    """ This tests that given the true model parameters, can it estimate the states correctly."""
    X = LineageTree.init_from_parameters(pi, T, E, desired_num_cells=(2 ** 7) - 1, desired_experimental_time=200, censor_condition=cens)

    tHMMobj = tHMM([X], num_states=2)  # build the tHMM class with X
    tHMMobj.estimate.pi = pi
    tHMMobj.estimate.T = T
    tHMMobj.estimate.E = E

    do_E_step(tHMMobj)
    pred_states = tHMMobj.predict()
    true_states = [cell.state for cell in tHMMobj.X[0].output_lineage]

    assert rand_score(true_states, pred_states[0]) >= 0.95

@pytest.mark.parametrize("cens", [0, 2])
def test_M_step(cens):
    X = LineageTree.init_from_parameters(pi, T, E, desired_num_cells=(2 ** 10) - 1, desired_experimental_time=400, censor_condition=cens)
    tHMMobj = tHMM([X], num_states=2) 
    gammas = [np.zeros((len(lineage.output_lineage), tHMMobj.num_states)) for lineage in tHMMobj.X]

    # create the gamma matrix (N x K) that shows the probability of a cell n being in state k from the true state assignments.
    for idx, g_lin in enumerate(gammas):
        for i in range(g_lin.shape[0]):
            g_lin[i, tHMMobj.X[idx].output_lineage[i].state] = 1

    do_M_E_step(tHMMobj, gammas)
    # test bernoulli
    np.testing.assert_allclose(tHMMobj.estimate.E[0].params[0], E[0].params[0], rtol=0.1)
    # gamma parameters
    np.testing.assert_allclose(tHMMobj.estimate.E[0].params[1:], E[0].params[1:], rtol=0.5)
