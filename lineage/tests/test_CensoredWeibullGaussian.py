"""Tests for the censored Weibull / Gaussian biosensor emission and its M step."""

import numpy as np
import pytest
import scipy.stats as sp
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.integrate import quad
from scipy.optimize import minimize

from ..Analyze import Analyze_list, fit_list
from ..LineageTree import LineageTree
from ..states.CensoredWeibullGaussian import StateDistribution, censor_lineage_weibull, split_obs
from ..states.weibullFit import (
    KAPPA_BOUNDS,
    MAX_SCALE_RATIO,
    gaussian_estimator,
    weibull_estimator,
    weibull_loglik,
    weibull_logterms,
)
from ..tHMM import tHMM


@pytest.fixture
def dist():
    return StateDistribution(mu=0.8, sigma=0.25, kappa=2.5, lam=30.0)


def censored_sample(rng, n, kappa, lam, cmax):
    """Weibull lifetimes right-censored by a uniform follow-up time."""
    t = lam * rng.weibull(kappa, n)
    c = rng.uniform(0.0, cmax, n)
    return np.minimum(t, c), (t <= c).astype(float)


# -- likelihood --------------------------------------------------------------


def test_logpdf_matches_scipy(dist):
    """Each case is the Gaussian term plus a Weibull density or survival term."""
    x = np.array(
        [
            [0.5, 20.0, 1.0],  # division observed
            [0.5, 20.0, 0.0],  # censored
            [np.nan, 20.0, 1.0],  # biosensor missing
            [0.5, np.nan, np.nan],  # lifetime missing
            [-0.5, -20.0, -1.0],  # hidden for cross validation
        ]
    )
    n = sp.norm(0.8, 0.25)
    w = sp.weibull_min(2.5, scale=30.0)
    expected = [
        n.logpdf(0.5) + w.logpdf(20.0),
        n.logpdf(0.5) + w.logsf(20.0),
        w.logpdf(20.0),
        n.logpdf(0.5),
        0.0,
    ]
    np.testing.assert_allclose(dist.logpdf(x), expected)


def test_survival_boundaries(dist):
    """S(0) = 1, S(lam) = 1/e, and S decreases monotonically to 0 as t -> inf."""
    t = np.array([0.0, 1e-12, 30.0, 300.0, 1e4])
    x = np.column_stack([np.full(t.size, np.nan), t, np.zeros(t.size)])
    logS = dist.logpdf(x)

    assert logS[0] == pytest.approx(0.0, abs=1e-20)
    assert logS[1] == pytest.approx(0.0, abs=1e-20)
    assert logS[2] == pytest.approx(-1.0)
    assert np.all(np.diff(logS) <= 0.0) and logS[2] > logS[3] > logS[4]
    assert np.exp(logS[-1]) == 0.0


def test_density_at_zero_is_finite_for_any_shape():
    """A recorded lifetime of exactly zero never produces a non-finite emission, even for
    shapes below one where the density itself diverges at the origin."""
    for kappa in (0.5, 1.0, 3.0):
        d = StateDistribution(0.0, 1.0, kappa, 10.0)
        ll = d.logpdf(np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]))
        assert np.all(np.isfinite(ll))


def test_density_is_normalized_and_matches_survival(dist):
    """f integrates to one, and f = -dS/dt."""
    f = lambda t: np.exp(dist.logpdf(np.array([[np.nan, t, 1.0]]))[0])  # noqa: E731
    S = lambda t: np.exp(dist.logpdf(np.array([[np.nan, t, 0.0]]))[0])  # noqa: E731

    assert quad(f, 0.0, np.inf)[0] == pytest.approx(1.0, abs=1e-8)
    for t in (5.0, 30.0, 60.0):
        h = 1e-5
        assert f(t) == pytest.approx(-(S(t + h) - S(t - h)) / (2 * h), rel=1e-5)


def test_split_obs():
    x = np.array([[1.0, 5.0, 1.0], [1.0, 5.0, 0.0], [np.nan, 5.0, 1.0], [-1.0, -5.0, -1.0], [1.0, np.nan, np.nan]])
    has_x, uncen, cen = split_obs(x)
    assert np.array_equal(has_x, [True, True, False, False, True])
    assert np.array_equal(uncen, [True, False, True, False, False])
    assert np.array_equal(cen, [False, True, False, False, False])


def test_rvs_moments(dist):
    x, t, delta = dist.rvs(200000, rng=0)
    assert np.mean(x) == pytest.approx(0.8, abs=0.01)
    assert np.std(x) == pytest.approx(0.25, abs=0.01)
    assert np.mean(t) == pytest.approx(dist.mean_lifetime(), rel=0.01)
    assert np.all(delta == 1.0)


def test_dist_is_a_metric_on_examples(dist):
    other = StateDistribution(0.1, 0.25, 1.5, 100.0)
    assert dist.dist(dist) == 0.0
    assert dist.dist(other) == pytest.approx(other.dist(dist))
    assert dist.dist(other) > 0.0


# -- Weibull M step ------------------------------------------------------------


def test_weibull_uncensored_matches_scipy():
    rng = np.random.default_rng(0)
    t = 25.0 * rng.weibull(1.7, 500)
    kappa, lam = weibull_estimator(t, np.ones_like(t), np.ones_like(t))
    k_sp, _, l_sp = sp.weibull_min.fit(t, floc=0.0)
    assert kappa == pytest.approx(k_sp, rel=1e-5)
    assert lam == pytest.approx(l_sp, rel=1e-5)


@pytest.mark.parametrize("seed", range(4))
def test_weibull_censored_weighted_is_the_mle(seed):
    """The Newton solution matches a general-purpose optimizer of the censored,
    weighted log likelihood."""
    rng = np.random.default_rng(seed)
    t, d = censored_sample(rng, 400, kappa=rng.uniform(0.6, 5.0), lam=rng.uniform(5, 50), cmax=60.0)
    w = rng.uniform(0.0, 1.0, t.size)

    kappa, lam = weibull_estimator(t, d, w)
    res = minimize(
        lambda p: -weibull_loglik(t, d, w, np.exp(p[0]), np.exp(p[1])),
        x0=[0.0, np.log(np.mean(t))],
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000},
    )
    assert weibull_loglik(t, d, w, kappa, lam) >= -res.fun - 1e-8
    np.testing.assert_allclose([kappa, lam], np.exp(res.x), rtol=1e-4)


def test_weibull_score_vanishes_at_estimate():
    rng = np.random.default_rng(1)
    t, d = censored_sample(rng, 300, 2.0, 20.0, 40.0)
    w = rng.uniform(0.1, 1.0, t.size)
    kappa, lam = weibull_estimator(t, d, w)
    h = 1e-6
    for dk, dl in ((h, 0.0), (0.0, h)):
        up = weibull_loglik(t, d, w, kappa * np.exp(dk), lam * np.exp(dl))
        dn = weibull_loglik(t, d, w, kappa * np.exp(-dk), lam * np.exp(-dl))
        assert (up - dn) / (2 * h) == pytest.approx(0.0, abs=1e-4)


def test_weibull_recovers_parameters_under_heavy_censoring():
    """With ~60% of lifetimes censored the estimator is still consistent."""
    rng = np.random.default_rng(2)
    t, d = censored_sample(rng, 50000, 2.5, 40.0, 50.0)
    assert 0.5 < 1 - d.mean() < 0.7
    kappa, lam = weibull_estimator(t, d, np.ones_like(t))
    assert kappa == pytest.approx(2.5, rel=0.03)
    assert lam == pytest.approx(40.0, rel=0.03)


def test_weibull_integer_weights_equal_repeats():
    rng = np.random.default_rng(3)
    t, d = censored_sample(rng, 100, 1.5, 10.0, 20.0)
    reps = rng.integers(1, 4, t.size)
    a = weibull_estimator(t, d, reps.astype(float))
    b = weibull_estimator(np.repeat(t, reps), np.repeat(d, reps), np.ones(reps.sum()))
    np.testing.assert_allclose(a, b, rtol=1e-8)


def test_weibull_zero_weights_are_ignored():
    rng = np.random.default_rng(4)
    t, d = censored_sample(rng, 200, 1.5, 10.0, 20.0)
    w = np.ones_like(t)
    w[:50] = 0.0
    np.testing.assert_allclose(weibull_estimator(t, d, w), weibull_estimator(t[50:], d[50:], w[50:]), rtol=1e-10)


def test_weibull_no_events_caps_scale():
    """A state in which no cell ever divides has an infinite MLE scale; it is capped."""
    t = np.array([10.0, 20.0, 30.0])
    kappa, lam = weibull_estimator(t, np.zeros(3), np.ones(3), kappa0=2.0)
    assert kappa == 2.0
    assert lam == MAX_SCALE_RATIO * 30.0


def test_weibull_identical_event_times_hits_shape_bound():
    t = np.full(20, 12.0)
    kappa, lam = weibull_estimator(t, np.ones(20), np.ones(20))
    assert kappa == KAPPA_BOUNDS[1]
    assert lam == pytest.approx(12.0, rel=0.01)


@settings(max_examples=40, deadline=None)
@given(
    kappa=st.floats(0.3, 8.0),
    lam=st.floats(0.5, 200.0),
    scale=st.floats(1e-3, 1e3),
    seed=st.integers(0, 2**32 - 1),
)
def test_weibull_estimate_is_scale_equivariant_and_optimal(kappa, lam, scale, seed):
    """Rescaling time rescales the fitted scale and leaves the shape alone, and the
    estimate beats nearby parameter values."""
    rng = np.random.default_rng(seed)
    t, d = censored_sample(rng, 60, kappa, lam, 2 * lam)
    # Durations are floored at 1e-10 before taking logs, which is not scale equivariant.
    if d.sum() < 2 or np.unique(t[d == 1]).size < 2 or min(t.min(), t.min() * scale) < 1e-8:
        return
    w = rng.uniform(0.05, 1.0, t.size)

    k1, l1 = weibull_estimator(t, d, w)
    k2, l2 = weibull_estimator(t * scale, d, w)
    assert k2 == pytest.approx(k1, rel=1e-6)
    assert l2 == pytest.approx(l1 * scale, rel=1e-6)

    if KAPPA_BOUNDS[0] < k1 < KAPPA_BOUNDS[1]:
        best = weibull_loglik(t, d, w, k1, l1)
        for dk, dl in ((1.02, 1.0), (0.98, 1.0), (1.0, 1.02), (1.0, 0.98)):
            assert weibull_loglik(t, d, w, k1 * dk, l1 * dl) <= best + 1e-9


def test_gaussian_estimator():
    x = np.array([1.0, 2.0, 3.0, 10.0])
    w = np.array([1.0, 2.0, 1.0, 0.0])
    mu, sigma = gaussian_estimator(x, w, min_sigma=1e-3)
    assert mu == pytest.approx(2.0)
    assert sigma == pytest.approx(np.sqrt(0.5))
    assert gaussian_estimator(np.ones(3), np.ones(3), min_sigma=0.1) == (1.0, 0.1)


def test_estimator_recovers_parameters(dist):
    """The single-state M step fits both halves of the emission."""
    rng = np.random.default_rng(5)
    x, t, _ = dist.rvs(20000, rng=rng)
    c = rng.uniform(0.0, 60.0, t.size)
    obs = np.column_stack([x, np.minimum(t, c), (t <= c).astype(float)])
    obs[::10, 0] = np.nan  # some cells have no biosensor reading

    fit = StateDistribution()
    fit.estimator(obs, np.ones(t.size))
    np.testing.assert_allclose(fit.params, dist.params, rtol=0.03)


# -- truncation ------------------------------------------------------------------


def truncated_sample(rng, n, kappa, lam, width, cmax=np.inf):
    """Lifetimes only seen when they end inside a random window ``(a, a + width)``,
    optionally right-censored at a follow-up time ``c`` below the window's end."""
    t = lam * rng.weibull(kappa, 20 * n)
    a = rng.uniform(0.0, lam, t.size)
    keep = (t > a) & (t < a + width)
    t, a = t[keep][:n], a[keep][:n]
    b = a + width
    c = np.minimum(a + rng.uniform(0.0, cmax, t.size), b) if np.isfinite(cmax) else b
    return np.minimum(t, c), (t <= c).astype(float), a, b


def test_truncated_logterms_match_scipy(dist):
    """Events are f(t) / (S(a) - S(b)); censored cells are (S(t) - S(b)) / (S(a) - S(b))."""
    w = sp.weibull_min(2.5, scale=30.0)
    t = np.array([20.0, 20.0, 20.0, 20.0, 35.0])
    ev = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    lo = np.array([10.0, 10.0, 0.0, 0.0, 0.0])
    hi = np.array([40.0, 40.0, 25.0, np.inf, np.inf])
    den = np.log(w.sf(lo) - w.sf(hi))
    num = np.where(ev == 1, w.logpdf(t), np.log(w.sf(t) - w.sf(hi)))
    val, _ = weibull_logterms(t, ev, lo, hi, 2.5, 30.0)
    np.testing.assert_allclose(val, num - den, rtol=1e-12)
    # No window is the plain censored likelihood.
    assert val[3] == pytest.approx(w.logsf(20.0))
    assert val[4] == pytest.approx(w.logpdf(35.0))

    # The emission reads the window from columns 3-4, NaN meaning no bound.
    obs = np.column_stack([np.full(5, np.nan), t, ev, np.where(lo > 0, lo, np.nan), hi])
    np.testing.assert_allclose(dist.logpdf(obs), val, rtol=1e-12)
    np.testing.assert_allclose(dist.logpdf(obs[3:, :3]), val[3:], rtol=1e-12)


def test_truncated_density_is_normalized():
    def f(t):
        return np.exp(weibull_logterms(np.array([t]), np.ones(1), np.array([8.0]), np.array([30.0]), 1.7, 20.0)[0][0])

    assert quad(f, 8.0, 30.0)[0] == pytest.approx(1.0, abs=1e-9)


def test_truncated_gradient_matches_finite_differences():
    rng = np.random.default_rng(7)
    t, d, a, b = truncated_sample(rng, 50, 2.0, 20.0, 15.0, cmax=10.0)
    b[::3] = np.inf
    a[::4] = 0.0
    _, grad = weibull_logterms(t, d, a, b, 2.0, 20.0)
    h = 1e-6
    for j, (dk, dl) in enumerate(((h, 0.0), (0.0, h))):
        up = weibull_logterms(t, d, a, b, 2.0 * np.exp(dk), 20.0 * np.exp(dl))[0]
        dn = weibull_logterms(t, d, a, b, 2.0 * np.exp(-dk), 20.0 * np.exp(-dl))[0]
        np.testing.assert_allclose(grad[:, j], (up - dn) / (2 * h), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("seed", range(3))
def test_truncated_estimator_is_the_mle(seed):
    """The numerical fit is a stationary point that beats a general optimizer from several starts."""
    rng = np.random.default_rng(seed)
    t, d, a, b = truncated_sample(rng, 300, rng.uniform(1.0, 4.0), rng.uniform(10, 40), 20.0, cmax=15.0)
    w = rng.uniform(0.0, 1.0, t.size)
    kappa, lam = weibull_estimator(t, d, w, lo=a, hi=b)

    def nll(p):
        with np.errstate(all="ignore"):
            v = -np.dot(w, weibull_logterms(t, d, a, b, np.exp(p[0]), np.exp(p[1]))[0])
        return v if np.isfinite(v) else np.inf

    best = -nll(np.log([kappa, lam]))
    for x0 in ([0.0, np.log(np.mean(t))], [1.0, np.log(np.max(t))], [np.log(kappa), np.log(lam)]):
        res = minimize(nll, x0=x0, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12})
        assert best >= -res.fun - 1e-6
    grad = w @ weibull_logterms(t, d, a, b, kappa, lam)[1]
    np.testing.assert_allclose(grad / w.sum(), 0.0, atol=1e-4)


def test_truncated_estimator_removes_selection_bias():
    """Lifetimes selected to end inside a short window are biased toward it; conditioning on
    the window recovers the generating distribution, and the naive fit does not."""
    rng = np.random.default_rng(8)
    t, d, a, b = truncated_sample(rng, 20000, 2.0, 30.0, 15.0)
    naive = weibull_estimator(t, d, np.ones_like(t))
    kappa, lam = weibull_estimator(t, d, np.ones_like(t), lo=a, hi=b)
    assert kappa == pytest.approx(2.0, rel=0.05)
    assert lam == pytest.approx(30.0, rel=0.05)
    assert abs(naive[0] - 2.0) > 0.3


def test_truncated_events_do_not_force_a_slow_state_to_divide():
    """For a state far slower than the window, f(t) / (S(a) - S(b)) no longer depends on
    the scale, so cells that divided only because they were selected on it leave the scale
    free to stay long, rather than pulling it down to the window as the naive fit does."""
    rng = np.random.default_rng(9)
    a = rng.uniform(0.0, 20.0, 200)
    b = a + 20.0
    t = rng.uniform(a, b)
    lam_trunc = weibull_estimator(t, np.ones_like(t), np.ones_like(t), kappa0=3.0, lo=a, hi=b)[1]
    lam_naive = weibull_estimator(t, np.ones_like(t), np.ones_like(t))[1]
    assert lam_naive < 30.0
    assert lam_trunc > 100.0


def test_six_column_observations_are_not_split_into_phases():
    """Truncation columns plus one extra make six, which the Gamma model reads as two
    phases; this emission fits whole rows whatever their width."""
    rng = np.random.default_rng(10)
    E = [StateDistribution(0.0, 0.3, 2.0, 40.0), StateDistribution(1.0, 0.3, 4.0, 20.0)]
    pops = []
    for _ in range(2):
        pop = []
        for _ in range(10):
            lin = LineageTree.rand_init(np.array([0.5, 0.5]), np.full((2, 2), 0.5), E, 15, 0, rng=rng)
            obs = np.column_stack([lin.obs, np.full((len(lin), 2), np.nan), np.zeros(len(lin))])
            pop.append(LineageTree(lin.tree, E, obs=obs))
        pops.append(pop)
    objs = [tHMM(X, num_states=2, rng=rng) for X in pops]
    fit_list(objs, rng=rng, shared_T=False, estimate_pi=True)
    means = sorted(e.params[0] for e in objs[0].estimate.E)
    np.testing.assert_allclose(means, [0.0, 1.0], atol=0.15)


# -- lineage censoring ------------------------------------------------------------


def test_censor_lineage_weibull():
    E = [StateDistribution(0.0, 1.0, 3.0, 20.0)]
    full = LineageTree.rand_init(np.ones(1), np.ones((1, 1)), E, 2**8 - 1, censor_condition=0, rng=0)
    end = 50.0
    tree, obs, _ = censor_lineage_weibull(full.tree, full.obs, full.states, 2, end)
    lin = LineageTree(tree, E, obs=obs)

    parents, daughters = lin.edges
    start = np.zeros(len(lin))
    for p, d in zip(parents, daughters, strict=True):
        start[d] = start[p] + obs[p, 1]
    stop = start + obs[:, 1]

    assert np.all(start < end)
    # Censored cells were running at the end; divided ones finished before it.
    np.testing.assert_allclose(stop[obs[:, 2] == 0], end)
    assert np.all(stop[obs[:, 2] == 1] <= end)
    # A censored cell never divided, so it has no daughters; every divided cell has two.
    n_children = np.diff(tree.indptr)
    assert np.all(n_children[obs[:, 2] == 0] == 0)
    assert np.all(n_children[obs[:, 2] == 1] == 2)
    # The biosensor is untouched.
    assert np.all(np.isfinite(obs[:, 0]))
    assert 0 < np.sum(obs[:, 2] == 0) < len(lin)


# -- end to end ----------------------------------------------------------------


def test_tHMM_recovers_states_and_transitions():
    """A two-state tHMM fit to censored lineages finds the generating model."""
    rng = np.random.default_rng(6)
    T = np.array([[0.85, 0.15], [0.1, 0.9]])
    E = [StateDistribution(0.2, 0.3, 2.0, 60.0), StateDistribution(1.5, 0.3, 4.0, 20.0)]
    pop = [LineageTree.rand_init(np.array([0.5, 0.5]), T, E, 63, 2, 110.0, rng=rng) for _ in range(60)]

    [tO], _, _ = Analyze_list([pop], 2, rng=rng)
    order = np.argsort([e.params[0] for e in tO.estimate.E])
    est_T = tO.estimate.T[np.ix_(order, order)]
    est_E = [tO.estimate.E[i] for i in order]

    np.testing.assert_allclose(est_T, T, atol=0.1)
    for fit, true in zip(est_E, E, strict=True):
        assert fit.params[0] == pytest.approx(true.params[0], abs=0.05)
        assert fit.params[1] == pytest.approx(true.params[1], abs=0.05)
        assert fit.params[2] == pytest.approx(true.params[2], rel=0.25)
        assert fit.params[3] == pytest.approx(true.params[3], rel=0.15)
