"""Tests for mini-batch (online) Baum-Welch."""

from copy import deepcopy

import numpy as np
import pytest

from ..Analyze import Analyze_list
from ..BaumWelch import calculate_log_likelihood, calculate_stationary, do_E_step
from ..LineageTree import LineageTree
from ..states.StateDistributionGamma import StateDistribution as GammaSD
from ..states.StateDistributionGaPhs import StateDistribution as PhaseSD
from ..StochasticEM import _make_batches, _revive_states, cosine_schedule, fit_list_stochastic
from ..tHMM import tHMM

T = np.array([[0.9, 0.1], [0.1, 0.9]])
pi = calculate_stationary(T)
E_gamma = [GammaSD(0.99, 20, 0.5), GammaSD(0.85, 10, 2.5)]
E_phase = [PhaseSD(0.99, 0.9, 20, 0.5, 20, 1.0), PhaseSD(0.88, 0.75, 10, 1.5, 10, 2.5)]


def make_population(E, n_lineages=20, censor=0, seed=0):
    rng = np.random.default_rng(seed)
    return [
        LineageTree.rand_init(
            pi, T, E, desired_num_cells=15, censor_condition=censor, desired_experiment_time=100, rng=rng
        )
        for _ in range(n_lineages)
    ]


def test_cosine_schedule():
    """The step size anneals monotonically from alpha0 to final_ratio * alpha0."""
    a = cosine_schedule(0.5, 20, final_ratio=0.1)
    assert a.size == 20
    assert a[0] == pytest.approx(0.5)
    assert a[-1] == pytest.approx(0.05)
    assert np.all(np.diff(a) <= 0)


def test_make_batches_partition():
    """Each condition's lineages appear in exactly one batch per epoch, in every batch."""
    batches = _make_batches([10, 7, 3], 4, np.random.default_rng(0))
    assert len(batches) == 4
    for c, n in enumerate([10, 7, 3]):
        idx = np.concatenate([b[c] for b in batches])
        np.testing.assert_array_equal(np.sort(idx), np.arange(n))
    # The two larger conditions are represented in every batch
    assert all(b[0].size > 0 and b[1].size > 0 for b in batches)


@pytest.mark.parametrize("censor", [0, 3])
def test_fit_single(censor):
    """A stochastic fit improves on the initial model and returns a self-consistent fit."""
    tHMMobj = tHMM(make_population(E_phase, censor=censor), num_states=2, rng=1)
    _, NF0, _, _ = do_E_step(tHMMobj)
    LL0 = calculate_log_likelihood(NF0)

    NF, gammas, LL = fit_list_stochastic([tHMMobj], epochs=3, polish_iter=2, rng=1)
    assert np.isfinite(LL)
    assert LL > LL0
    assert LL == pytest.approx(calculate_log_likelihood(NF))

    # The returned likelihood belongs to the parameters left on the object
    _, NF_check, _, gammas_check = do_E_step(tHMMobj)
    assert calculate_log_likelihood(NF_check) == pytest.approx(LL)
    for g, gc in zip(gammas[0], gammas_check, strict=True):
        np.testing.assert_allclose(g, gc)

    np.testing.assert_allclose(tHMMobj.estimate.T.sum(axis=1), 1.0)
    np.testing.assert_allclose(tHMMobj.estimate.pi.sum(), 1.0)


def test_fit_atonce():
    """Fitting several conditions at once shares T and pi and gives a finite fit."""
    pops = [make_population(E_gamma, seed=s, censor=3) for s in range(3)]
    objs = [tHMM(X, num_states=2, rng=2) for X in pops]
    _, gammas, LL = fit_list_stochastic(objs, epochs=2, polish_iter=1, rng=2)

    assert np.isfinite(LL)
    assert len(gammas) == 3
    for o in objs[1:]:
        np.testing.assert_array_equal(o.estimate.T, objs[0].estimate.T)
        np.testing.assert_array_equal(o.estimate.pi, objs[0].estimate.pi)


def test_fit_seed():
    """The stochastic fit is reproducible given a seed."""
    tHMMobj = tHMM(make_population(E_gamma), num_states=2, rng=3)
    _, _, LL1 = fit_list_stochastic([deepcopy(tHMMobj)], epochs=2, polish_iter=1, rng=4)
    _, _, LL2 = fit_list_stochastic([deepcopy(tHMMobj)], epochs=2, polish_iter=1, rng=4)
    assert LL1 == LL2


def test_revive_states():
    """A starved state gets occupancy back from a donor, and the T statistics stay consistent."""
    tHMMobj = tHMM(make_population(E_gamma), num_states=3, rng=5)
    rng = np.random.default_rng(5)
    weights = [[rng.dirichlet(np.ones(3), size=len(lin)) for lin in tHMMobj.X]]
    for w in weights[0]:
        w[:, 2] = 0.0  # state 2 has collapsed

    numer = rng.uniform(1.0, 2.0, size=(3, 3))
    numer[2, :] = numer[:, 2] = 0.0
    denom = numer.sum(axis=1)
    total_weight = np.vstack(weights[0]).sum()

    assert _revive_states([tHMMobj], weights, numer, denom, threshold=0.05, rng=rng) == 1

    occupancy = np.vstack(weights[0]).sum(axis=0)
    assert occupancy[2] > 0.05 * occupancy.sum() / 3
    assert np.vstack(weights[0]).sum() == pytest.approx(total_weight)
    assert np.all(np.vstack(weights[0]) >= 0.0)
    # Every row still sums to its denominator, so T stays a valid stochastic matrix
    np.testing.assert_allclose(numer.sum(axis=1), denom)
    assert numer[2].sum() > 0 and numer[:, 2].sum() > 0


def test_analyze_list_stochastic():
    """Analyze_list dispatches to the stochastic fit and keeps the best restart."""
    pop = make_population(E_gamma, censor=3)
    objs, LL, gammas = Analyze_list([pop], 2, rng=6, method="stochastic", n_init=2, epochs=2, polish_iter=1)
    assert np.isfinite(LL)
    assert len(objs) == 1 and len(gammas) == 1
