""" Unit test file. """
import pytest
import numpy as np
from sklearn.metrics import rand_score
from ..BaumWelch import do_E_step, do_M_T_step, do_M_pi_step, do_M_E_step, calculate_log_likelihood
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
    """ This tests that given the true cell states, can the model estimate parameters correctly. """
    X = LineageTree.init_from_parameters(pi, T, E, desired_num_cells=(2 ** 7) - 1, desired_experimental_time=200, censor_condition=cens)

    tHMMobj = tHMM([X], num_states=2)
    MSD, _, betas, gammas = do_E_step(tHMMobj)

    T_est = do_M_T_step([tHMMobj], [MSD], [betas], [gammas])
    pi_est = do_M_pi_step([tHMMobj], [gammas])
    do_M_E_step(tHMMobj, gammas)

    assert E_est[0].dist(E[0]) < 5.0
    assert E_est[1].dist(E[1]) < 5.0

    assert np.allclose(pi_est, pi)
    assert np.allclose(T_est, T)

