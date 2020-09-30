""" Unit test file. """
import pytest
import numpy as np
from ..BaumWelch import do_E_step, calculate_log_likelihood
from ..LineageTree import LineageTree
from ..tHMM import tHMM
from ..figures.figureCommon import pi, T, E


@pytest.mark.parametrize("cens", [0, 2])
@pytest.mark.parametrize("nStates", [1, 2, 3])
def test_BW(cens, nStates):
    """ This tests that one step of Baum-Welch increases the likelihood of the fit. """
    X = LineageTree.init_from_parameters(pi, T, E, desired_num_cells=(2 ** 7) - 1, desired_experimental_time=500, censor_condition=cens)
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
