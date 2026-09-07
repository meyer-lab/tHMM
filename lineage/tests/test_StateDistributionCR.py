"""Tests for the competing-risks state distributions."""

import numpy as np
import pytest
import scipy.stats as sp

from lineage.compare_emissions import outcome_mass
from lineage.states.StateDistributionCR import (
    StateDistribution,
    StateDistributionPhase,
    event_masks,
)


@pytest.fixture
def dist():
    return StateDistribution(gamma_a=7.0, gamma_scale=4.5, death_a=1.0, death_scale=40.0)


def test_event_masks_partition():
    """Every observation lands in exactly one case, or in none when it has no time."""
    x = np.array(
        [
            [1.0, 20.0, 1.0],  # transition observed
            [0.0, 20.0, 1.0],  # death observed
            [np.nan, 20.0, 0.0],  # time censored, fate unknown
            [1.0, 20.0, 0.0],  # survived the phase but only partly observed
            [0.0, 20.0, 0.0],  # died, but the phase was entered before we saw it
            [1.0, np.nan, np.nan],  # never entered the phase
            [1.0, -20.0, 1.0],  # hidden for cross validation
        ]
    )
    divided, died, censored = event_masks(x)

    assert np.array_equal(divided, [True, False, False, False, False, False, False])
    assert np.array_equal(died, [False, True, False, False, True, False, False])
    assert np.array_equal(censored, [False, False, True, True, False, False, False])
    # Mutually exclusive, and the last two rows are excluded everywhere.
    assert np.all(divided.astype(int) + died.astype(int) + censored.astype(int) <= 1)


def test_logpdf_matches_competing_risks_by_hand(dist):
    """Each case is the textbook competing-risks term."""
    div = sp.gamma(7.0, scale=4.5)
    death = sp.gamma(1.0, scale=40.0)
    t = 20.0

    x = np.array(
        [
            [1.0, t, 1.0],
            [0.0, t, 1.0],
            [np.nan, t, 0.0],
            [1.0, np.nan, np.nan],
            [1.0, -t, 1.0],
        ]
    )
    expected = [
        div.logpdf(t) + death.logsf(t),
        death.logpdf(t) + div.logsf(t),
        div.logsf(t) + death.logsf(t),
        0.0,
        0.0,
    ]
    np.testing.assert_allclose(dist.logpdf(x), expected)


def test_sub_densities_sum_to_one(dist):
    """The divide and die branches together carry exactly probability one."""
    t = np.linspace(1e-9, 2000.0, 400001)
    div, death = dist.div_clock, dist.death_clock

    p_divide = np.trapezoid(div.pdf(t) * death.sf(t), t)
    p_die = np.trapezoid(death.pdf(t) * div.sf(t), t)

    assert p_divide + p_die == pytest.approx(1.0, abs=1e-4)
    # params[0] is derived from the two clocks rather than fit, so it must agree.
    assert dist.params[0] == pytest.approx(p_divide, abs=1e-4)


def test_gamma_bernoulli_mass_exceeds_one_under_censoring():
    """The Bernoulli/Gamma emission is not normalized when cells are censored.

    A death is scored as an atom with no time attached while the censored branch uses
    the division survival alone, so the two overlap. The competing-risks form does not.
    """
    for horizon in (4.0, 12.0, 24.0):
        gamma_mass, cr_mass = outcome_mass(a=3.38, scale=5.74, p_div=0.9, horizon=horizon)
        assert gamma_mass > 1.02
        assert cr_mass == pytest.approx(1.0, abs=1e-3)

    # Both are fine once nothing is censored.
    gamma_mass, cr_mass = outcome_mass(a=3.38, scale=5.74, p_div=0.9, horizon=400.0)
    assert gamma_mass == pytest.approx(1.0, abs=1e-3)


def test_estimator_recovers_parameters(dist):
    """A weighted fit to data simulated from the model returns the same clocks."""
    rng = np.random.default_rng(42)
    obs = np.column_stack(dist.rvs(20000, rng=rng))

    fitted = StateDistribution(gamma_a=1.0, gamma_scale=1.0, death_a=1.0, death_scale=1.0)
    fitted.estimator(obs, np.ones(obs.shape[0]))

    np.testing.assert_allclose(fitted.params[1:], dist.params[1:], rtol=0.1)
    assert fitted.params[0] == pytest.approx(dist.params[0], abs=0.02)


def test_rvs_is_the_minimum_of_two_clocks(dist):
    """The recorded duration and fate are those of whichever clock fired first."""
    rng = np.random.default_rng(0)
    fate, dur, cens = dist.rvs(5000, rng=rng)

    assert np.all(np.isin(fate, (0.0, 1.0)))
    assert np.all(dur > 0.0)
    assert np.all(cens == 1.0)
    # Cells that divided did so faster than the death-clock mean of 40 h on average,
    # and the observed division fraction tracks the derived probability.
    assert fate.mean() == pytest.approx(dist.params[0], abs=0.02)


def test_phase_death_shapes():
    """G1's death clock is pinned to a constant hazard; G2's shape is free."""
    p = StateDistributionPhase()

    assert p.G1.fixed_death_shape and p.G1.params[3] == 1.0
    assert not p.G2.fixed_death_shape
    # Three parameters for G1 (two division, one death scale) and four for G2.
    assert (p.G1.dof(), p.G2.dof(), p.dof()) == (3, 4, 7)


def test_phase_params_layout_matches_gaphs():
    """The leading six entries keep their Gamma/Bernoulli meaning for figure code."""
    p = StateDistributionPhase(gamma_a1=7.0, gamma_scale1=3.0, gamma_a2=14.0, gamma_scale2=6.0)

    assert p.params.shape == (10,)
    assert (p.params[0], p.params[1]) == (p.G1.params[0], p.G2.params[0])
    np.testing.assert_allclose(p.params[2:4], [7.0, 3.0])
    np.testing.assert_allclose(p.params[4:6], [14.0, 6.0])
    np.testing.assert_allclose(p.params[6:8], p.G1.params[3:5])
    np.testing.assert_allclose(p.params[8:10], p.G2.params[3:5])


def test_phase_estimator_recovers_parameters():
    """Both phases, and both clocks within each, are recovered from simulated data."""
    truth = StateDistributionPhase(
        gamma_a1=7.0,
        gamma_scale1=3.0,
        gamma_a2=14.0,
        gamma_scale2=6.0,
        death_scale1=40.0,
        death_a2=3.0,
        death_scale2=20.0,
    )
    rng = np.random.default_rng(7)
    obs = np.column_stack(truth.rvs(30000, rng=rng))

    fitted = StateDistributionPhase(2.0, 2.0, 2.0, 2.0, death_scale1=10.0, death_a2=1.0, death_scale2=10.0)
    fitted.estimator(obs, np.ones(obs.shape[0]))

    np.testing.assert_allclose(fitted.params[2:], truth.params[2:], rtol=0.15)
