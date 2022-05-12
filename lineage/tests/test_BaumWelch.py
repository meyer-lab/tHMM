""" Unit test file. """
import pytest
import numpy as np
import pickle
from sklearn.metrics import rand_score
from ..BaumWelch import do_E_step, do_M_E_step, calculate_log_likelihood
from ..LineageTree import LineageTree
from ..tHMM import tHMM
from ..figures.common import pi, T, E, E2


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


pik1 = open("lapatinibs.pkl", "rb")
lpt = []
for i in range(4):
    lpt.append(pickle.load(pik1))

# model parameters for lapatinib 25nM
E3 = lpt[1].estimate.E
T3 = lpt[1].estimate.T
pi3 = lpt[1].estimate.pi

@pytest.mark.parametrize("cens", [0, 3])
def test_E_step(cens):
    """ This tests that given the true model parameters, can it estimate the states correctly."""
    population = []
    for _ in range(100):
        # make sure we have enough cells in the lineage.
        X = LineageTree.init_from_parameters(pi3, T3, E3, desired_num_cells=(2 ** 8) - 1, desired_experimental_time=300, censor_condition=cens)
        while len(X.output_lineage) < 9:
            X = LineageTree.init_from_parameters(pi3, T3, E3, desired_num_cells=(2 ** 8) - 1, desired_experimental_time=300, censor_condition=cens)
        population.append(X)

    tHMMobj = tHMM(population, num_states=lpt[1].num_states)  # build the tHMM class with X
    tHMMobj.estimate.pi = pi3
    tHMMobj.estimate.T = T3
    tHMMobj.estimate.E = E3

    do_E_step(tHMMobj)
    pred_states = tHMMobj.predict()
    true_states = [cell.state for cell in tHMMobj.X[0].output_lineage]

    assert rand_score(true_states, pred_states[0]) >= 0.9


@pytest.mark.parametrize("cens", [0, 3])
def test_M_step(cens):
    """ The M step of the BW. check the emission parameters if the true states are given. """

    population = []
    for _ in range(50):
        # make sure we have enough cells in the lineage.
        X = LineageTree.init_from_parameters(pi3, T3, E3, desired_num_cells=(2 ** 5) - 1, desired_experimental_time=100, censor_condition=cens)
        while len(X.output_lineage) < 4:
            X = LineageTree.init_from_parameters(pi3, T3, E3, desired_num_cells=(2 ** 5) - 1, desired_experimental_time=100, censor_condition=cens)
        population.append(X)

    tHMMobj = tHMM(population, num_states=lpt[1].num_states)
    gammas = [np.zeros((len(lineage.output_lineage), tHMMobj.num_states)) for lineage in tHMMobj.X]

    # create the gamma matrix (N x K) that shows the probability of a cell n being in state k from the true state assignments.
    for idx, g_lin in enumerate(gammas):
        for i in range(g_lin.shape[0]):
            g_lin[i, tHMMobj.X[idx].output_lineage[i].state] = 1

    do_M_E_step(tHMMobj, gammas)
    # test bernoulli
    if len(E3[0].params) > 3:  # phase-specific case
        for i in range(lpt[1].num_states):
            np.testing.assert_allclose(tHMMobj.estimate.E[i].params[0:2], E3[i].params[0:2], rtol=0.1)
            # gamma parameters
            np.testing.assert_allclose(tHMMobj.estimate.E[i].params[2:], E3[i].params[2:], rtol=0.5)
    else:
        for i in range(lpt[1].num_states):
            np.testing.assert_allclose(tHMMobj.estimate.E[i].params[0], E3[i].params[0], rtol=0.1)
            # gamma parameters
            np.testing.assert_allclose(tHMMobj.estimate.E[i].params[1:], E3[i].params[1:], rtol=0.5)
