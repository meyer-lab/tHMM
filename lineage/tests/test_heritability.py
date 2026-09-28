"""Tests for per-condition transition fitting, heritability summaries, and early forecasting."""

import numpy as np
import pytest

from ..Analyze import Analyze_list
from ..BaumWelch import calculate_log_likelihood, calculate_stationary, do_E_step, do_M_E_step, do_M_step
from ..early_biomarker import auc_table, cross_validated_scores, daughter_outcome, thmm_forecast
from ..heritability import (
    commitment_trees,
    dose_sweep,
    lrt,
    memory_eigenvalue,
    memory_half_life,
    persistence_half_life,
)
from ..LineageTree import LineageTree
from ..states.CensoredWeibullGaussian import StateDistribution
from ..tHMM import tHMM

E = [StateDistribution(0.2, 0.3, 2.0, 150.0), StateDistribution(1.5, 0.3, 4.0, 20.0)]


def simulate(Ts, n_lineages=30, end=96.0, seed=0):
    rng = np.random.default_rng(seed)
    pi = np.array([0.5, 0.5])
    return [[LineageTree.rand_init(pi, T, E, 63, 2, end, rng=rng) for _ in range(n_lineages)] for T in Ts]


# -- closed-form summaries -------------------------------------------------------


def test_persistence_half_life():
    T = np.array([[0.5, 0.5], [0.25, 0.75]])
    assert persistence_half_life(T, 0) == pytest.approx(1.0)
    assert persistence_half_life(T, 1) == pytest.approx(np.log(0.5) / np.log(0.75))
    assert persistence_half_life(np.eye(2), 1) == np.inf
    assert persistence_half_life(np.array([[0.0, 1.0], [1.0, 0.0]]), 0) == 0.0
    # tau_1/2 generations of persistence leave exactly half of the lines in the state.
    assert T[1, 1] ** persistence_half_life(T, 1) == pytest.approx(0.5)


@pytest.mark.parametrize("a,b", [(0.9, 0.8), (0.3, 0.6), (0.5, 0.5), (0.1, 0.2)])
def test_memory_eigenvalue_two_states(a, b):
    T = np.array([[a, 1 - a], [1 - b, b]])
    assert memory_eigenvalue(T) == pytest.approx(a + b - 1)


def test_memory_is_zero_when_rows_match():
    """Identical rows: the daughter's state does not depend on the mother's, whatever T_kk is."""
    T = np.array([[0.2, 0.3, 0.5]] * 3)
    assert memory_eigenvalue(T) == pytest.approx(0.0, abs=1e-12)
    assert memory_half_life(T) == 0.0
    assert memory_half_life(np.eye(3)) == np.inf
    # state-state correlation g generations apart decays as lambda_2 ** g
    T = np.array([[0.9, 0.1], [0.3, 0.7]])
    g = 5
    Tg = np.linalg.matrix_power(T, g)
    assert Tg[0, 0] - Tg[1, 0] == pytest.approx(memory_eigenvalue(T) ** g)


def test_lrt():
    out = lrt(-100.0, -95.0, 2)
    assert out["statistic"] == pytest.approx(10.0)
    assert out["p"] == pytest.approx(np.exp(-5.0))
    assert lrt(-100.0, -100.1, 2)["p"] == 1.0


# -- transition-matrix variants in Baum-Welch ------------------------------------


def test_independent_T_has_identical_rows():
    pops = simulate([np.array([[0.8, 0.2], [0.2, 0.8]])], n_lineages=10)
    [tO], _, _ = Analyze_list(pops, 2, rng=1, independent_T=True)
    np.testing.assert_allclose(tO.estimate.T[0], tO.estimate.T[1])
    np.testing.assert_allclose(tO.estimate.T.sum(axis=1), 1.0)


def test_per_condition_T_with_shared_emissions():
    Ts = [np.array([[0.6, 0.4], [0.1, 0.9]]), np.array([[0.9, 0.1], [0.4, 0.6]])]
    pops = simulate(Ts, n_lineages=40, seed=2)

    objs, _, _ = Analyze_list(pops, 2, rng=3, shared_T=False)
    # Emissions are shared, transitions are not.
    for e0, e1 in zip(objs[0].estimate.E, objs[1].estimate.E, strict=True):
        np.testing.assert_array_equal(e0.params, e1.params)
    assert not np.allclose(objs[0].estimate.T, objs[1].estimate.T)

    objs_shared, _, _ = Analyze_list(pops, 2, rng=3, shared_T=True)
    np.testing.assert_array_equal(objs_shared[0].estimate.T, objs_shared[1].estimate.T)


def test_estimate_pi_em_is_monotone():
    """With a freely estimated root distribution every EM iteration raises the likelihood."""
    pops = simulate([np.array([[0.7, 0.3], [0.2, 0.8]])], n_lineages=15, seed=11)
    tO = tHMM(pops[0], 2, rng=0)
    rng = np.random.default_rng(0)
    do_M_E_step(tO, [rng.dirichlet([1, 1], size=len(lin)) for lin in tO.X])
    LLs = []
    for _ in range(25):
        MSD, NF, betas, gammas = do_E_step(tO)
        LLs.append(calculate_log_likelihood(NF))
        do_M_step([tO], [MSD], [betas], [gammas], estimate_pi=True)
    # The T and pi pseudocounts make this MAP rather than ML, so allow a hair of slack.
    assert np.all(np.diff(LLs) > -1e-3)
    # Roots of simulated lineages are drawn from pi = (0.5, 0.5), not from T's stationary law.
    assert not np.allclose(tO.estimate.pi, calculate_stationary(tO.estimate.T), atol=0.02)


# -- dose sweep -----------------------------------------------------------------


@pytest.fixture(scope="module")
def heritable_sweep():
    Ts = [np.array([[0.6, 0.4], [0.1, 0.9]]), np.array([[0.85, 0.15], [0.3, 0.7]])]
    pops = simulate(Ts, n_lineages=40, seed=4)
    return dose_sweep(pops, [0, 500], rng=5), pops, Ts


def test_dose_sweep_recovers_transitions(heritable_sweep):
    sweep, _, Ts = heritable_sweep
    # States are ordered slowest first, which is the order they were simulated in.
    assert sweep.per_dose[0].estimate.E[0].mean_lifetime() > sweep.per_dose[0].estimate.E[1].mean_lifetime()
    # The cycling state's row is well determined: its cells divide, so their daughters are seen.
    for est, true in zip(sweep.T, Ts, strict=True):
        assert est[1, 1] == pytest.approx(true[1, 1], abs=0.08)
    np.testing.assert_allclose(sweep.half_lives(), [persistence_half_life(T, 1) for T in sweep.T])


def test_dose_sweep_nesting_and_tests(heritable_sweep):
    sweep, _, _ = heritable_sweep
    # Each null is nested in the per-dose model, which is warm-started from both.
    assert sweep.LL["per_dose"] >= sweep.LL["shared"] - 1e-6
    assert sweep.LL["per_dose"] >= sweep.LL["independent"] - 1e-6
    assert sweep.lrt["heritability"]["p"] < 1e-4
    assert sweep.lrt["dose_dependence"]["p"] < 0.01
    assert sweep.lrt["heritability"]["df"] == 2
    assert sweep.lrt["dose_dependence"]["df"] == 2


def test_dose_sweep_does_not_reject_nonheritable():
    """When every row is the same, the heritability test should not fire."""
    T = np.array([[0.3, 0.7], [0.3, 0.7]])
    pops = simulate([T, T], n_lineages=30, seed=6)
    sweep = dose_sweep(pops, [0, 100], rng=7)
    assert sweep.lrt["heritability"]["p"] > 0.01
    assert sweep.lrt["dose_dependence"]["p"] > 0.01
    assert np.all(np.abs(sweep.memory()) < 0.25)


def test_commitment_trees_are_consistent_posteriors(heritable_sweep):
    sweep, _, _ = heritable_sweep
    tO = sweep.per_dose[0]
    _, _, _, gammas = do_E_step(tO)
    for c, g in zip(commitment_trees(tO), gammas, strict=True):
        pair = c["pair"]
        np.testing.assert_allclose(pair.sum(axis=(1, 2)), 1.0, atol=1e-8)
        # The pairwise posterior's margins are the single-cell posteriors (up to the
        # machine-epsilon clipping the E step applies to near-zero messages).
        np.testing.assert_allclose(pair.sum(axis=2), g[c["parents"]], atol=1e-6)
        np.testing.assert_allclose(pair.sum(axis=1), g[c["daughters"]], atol=1e-6)
        assert np.all((c["p_switch"] >= -1e-12) & (c["p_switch"] <= 1 + 1e-12))


# -- early biomarker --------------------------------------------------------------


def test_daughter_outcome():
    t = np.array([10.0, 60.0, 30.0, 60.0, 48.0])
    delta = np.array([1.0, 1.0, 0.0, 0.0, 1.0])
    np.testing.assert_array_equal(daughter_outcome(t, delta, 48.0), [1.0, 0.0, np.nan, 0.0, 1.0])
    # An explicit escape time overrides division, e.g. S-phase entry before a late division.
    escape = np.array([5.0, 20.0, 10.0, np.nan, np.nan])
    np.testing.assert_array_equal(daughter_outcome(t, delta, 48.0, escape), [1.0, 1.0, 1.0, 0.0, np.nan])


def test_thmm_forecast_is_a_probability(heritable_sweep):
    sweep, _, _ = heritable_sweep
    tO = sweep.per_dose[0]
    x = np.linspace(-1, 3, 50)
    p = thmm_forecast(tO, x, np.full(50, 20.0))
    assert np.all((p >= 0) & (p <= 1))
    # A mother that looks cycling (high sensor) makes an escaping daughter more likely,
    # since the cycling state is persistent.
    assert p[-1] > p[0]


def test_forecast_auc(heritable_sweep):
    sweep, pops, _ = heritable_sweep
    cv = cross_validated_scores(sweep, pops, horizon=48.0, n_folds=3, rng=8)
    assert np.all(np.isfinite(cv["tHMM"])) and np.all(np.isfinite(cv["logistic"]))
    table = auc_table(cv, n_boot=50, rng=9)
    for row in table.values():
        assert row["lo"] <= row["auc"] <= row["hi"]
    # The mother's observation carries her state, and states are heritable.
    assert table["tHMM"]["auc"] > 0.6
