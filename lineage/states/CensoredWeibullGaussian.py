"""Joint biosensor / right-censored lifetime emission.

Each cell carries the observation ``[x, t, delta]``:

* ``x`` -- a continuous biosensor readout (e.g. the maximum rate of CDK2 activation in
  G1), NaN when it could not be measured;
* ``t`` -- how long the cell was followed, from birth to division or to the end of
  tracking;
* ``delta`` -- 1 when the division was observed, 0 when the lifetime is right-censored
  (the cell was still undivided when imaging stopped, or it left the field of view,
  or it died).

Under hidden state ``k`` the two are conditionally independent,

.. math::

    P(x, t, \\delta \\mid z = k) = \\mathcal{N}(x \\mid \\mu_k, \\sigma_k^2)\\,
        f_W(t \\mid \\lambda_k, \\kappa_k)^{\\delta}\\, S_W(t \\mid \\lambda_k, \\kappa_k)^{1 - \\delta},

with :math:`S_W(t) = \\exp(-(t/\\lambda)^\\kappa)`.

Two optional columns, ``[x, t, delta, t_lo, t_hi]``, truncate the lifetime: the cell is
only in the data because its lifetime fell in ``(t_lo, t_hi)``, and the Weibull factor is
divided by :math:`S_W(t_{lo}) - S_W(t_{hi})` (see :mod:`.weibullFit`). NaN means no bound.
This is the case for lineage roots that were selected by dividing inside a time window.
Any further columns are carried along and ignored.

Treating death as censoring of the
division clock is the cause-specific-hazard reading of a competing risk: it fits the
division hazard correctly, but says nothing about the death hazard.

As elsewhere in this package, a negative duration marks an observation hidden for cross
validation; that cell contributes nothing to the likelihood or to the fit.
"""

import numpy as np
from scipy.sparse import csr_array
from scipy.special import gamma as gamma_fn

from .weibullFit import gaussian_estimator, weibull_estimator, weibull_logterms

LOG_2PI = np.log(2.0 * np.pi)

#: Observation columns of the optional lower and upper lifetime truncation bounds.
TRUNC_COLS = (3, 4)


def truncation_bounds(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row truncation window ``(lo, hi)``, ``(0, inf)`` where there is none."""
    n = x.shape[0]
    if x.shape[1] <= TRUNC_COLS[1]:
        return np.zeros(n), np.full(n, np.inf)
    lo, hi = x[:, TRUNC_COLS[0]], x[:, TRUNC_COLS[1]]
    return np.where(np.isfinite(lo), lo, 0.0), np.where(np.isfinite(hi), hi, np.inf)


def split_obs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Masks for (usable biosensor, uncensored lifetime, censored lifetime)."""
    sensor, t, delta = x[:, 0], x[:, 1], x[:, 2]
    timed = np.isfinite(t) & (t >= 0.0)
    has_x = np.isfinite(sensor) & ~(np.isfinite(t) & (t < 0.0))
    return has_x, timed & (delta == 1.0), timed & (delta != 1.0)


def censor_lineage_weibull(
    tree: csr_array,
    obs: np.ndarray,
    states: np.ndarray,
    censor_condition: int,
    desired_experiment_time: float = 2e12,
) -> tuple[csr_array, np.ndarray, np.ndarray]:
    """Truncate a simulated lineage at the end of the experiment.

    Any ``censor_condition`` other than 0 applies time censoring (there is no death in
    this emission, so fate censoring is a no-op). A cell alive at the end has its
    lifetime cut to the time it was followed and flagged as censored; its descendants,
    which were never born, are dropped. The biosensor reading is kept, since it is taken
    in G1, before the lifetime runs out.
    """
    if censor_condition == 0:
        return tree, obs, states

    n = tree.shape[0]
    obs = obs.copy()
    parents = np.repeat(np.arange(n), np.diff(tree.indptr))
    daughters = tree.indices

    start = np.zeros(n)
    # Edges are stored parent-first, so a single pass fills in every birth time.
    for p, d in zip(parents, daughters, strict=True):
        start[d] = start[p] + obs[p, 1]

    observed = start < desired_experiment_time
    running = observed & (start + obs[:, 1] > desired_experiment_time)
    obs[running, 1] = desired_experiment_time - start[running]
    obs[running, 2] = 0.0

    kept = np.nonzero(observed)[0]
    return tree[kept, :][:, kept], obs[kept, :], states[kept]


def fit_emission(dist, x: np.ndarray, weights: np.ndarray):
    """Weighted M step for one state, writing the fitted parameters into ``dist``."""
    has_x, uncen, cen = split_obs(x)
    timed = uncen | cen

    if np.sum(weights[has_x]) > 0.0:
        dist.params[0], dist.params[1] = gaussian_estimator(x[has_x, 0], weights[has_x], dist.min_sigma)

    if np.sum(weights[timed]) > 0.0:
        lo, hi = truncation_bounds(x[timed])
        dist.params[2], dist.params[3] = weibull_estimator(
            x[timed, 1],
            uncen[timed].astype(float),
            weights[timed],
            kappa0=dist.params[2],
            lo=lo,
            hi=hi,
            lam0=dist.params[3],
        )


def atonce_estimator(all_tHMMobj: list, x_list: list, gammas_list: list[np.ndarray], phase: str = "all"):
    """M step across several conditions at once.

    The emissions are shared between conditions, so that a hidden state names the same
    phenotype at every dose and only the transition matrices are free to differ. This is
    what makes per-condition persistence probabilities comparable.
    """
    assert phase == "all", "CensoredWeibullGaussian has no cell-cycle phases."
    x = np.concatenate([np.asarray(xx) for xx in x_list], axis=0)
    gms = np.concatenate(gammas_list, axis=0)

    for state_j in range(len(all_tHMMobj[0].estimate.E)):
        ref = all_tHMMobj[0].estimate.E[state_j]
        fit_emission(ref, x, gms[:, state_j])
        for tO in all_tHMMobj[1:]:
            tO.estimate.E[state_j].params[:] = ref.params


class StateDistribution:
    """Gaussian biosensor readout with a right-censored Weibull lifetime.

    ``params`` is ``[mu, sigma, kappa, lam]``: the biosensor mean and standard deviation,
    and the Weibull shape and scale of the time to division.
    """

    #: BaumWelch looks this up on the emission object to pick the right M step.
    atonce_estimator = staticmethod(atonce_estimator)
    #: Observation rows are never split into cell-cycle phases, whatever their width.
    split_phases = False

    def __init__(self, mu: float = 1.0, sigma: float = 0.3, kappa: float = 3.0, lam: float = 20.0, min_sigma=1e-3):
        """
        :param min_sigma: floor on the fitted biosensor standard deviation, which stops a
            state that captures a single cell from collapsing onto it.
        """
        assert sigma > 0 and kappa > 0 and lam > 0
        self.params = np.array([mu, sigma, kappa, lam], dtype=float)
        self.min_sigma = min_sigma

    def rvs(self, size: int, rng=None):
        """Draw ``size`` uncensored ``(x, t, delta)`` observations."""
        rng = np.random.default_rng(rng)
        x = rng.normal(self.params[0], self.params[1], size=size)
        t = self.params[3] * rng.weibull(self.params[2], size=size)
        return x, t, np.ones(size)

    def mean_lifetime(self) -> float:
        """Mean of the (uncensored) division time, :math:`\\lambda\\,\\Gamma(1 + 1/\\kappa)`."""
        return float(self.params[3] * gamma_fn(1.0 + 1.0 / self.params[2]))

    def dist(self, other) -> float:
        """Distance between two states: the 1-Wasserstein distance between their lifetime
        means plus the 2-Wasserstein distance between their biosensor Gaussians."""
        assert isinstance(self, type(other))
        w_life = abs(self.mean_lifetime() - other.mean_lifetime())
        w_sensor = np.hypot(self.params[0] - other.params[0], self.params[1] - other.params[1])
        return float(w_life + w_sensor)

    def dof(self) -> int:
        return 4

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """Log likelihood of each cell's observation under this state."""
        mu, sigma, kappa, lam = self.params
        has_x, uncen, cen = split_obs(x)

        ll = np.zeros(x.shape[0])
        z = (x[has_x, 0] - mu) / sigma
        ll[has_x] += -0.5 * (LOG_2PI + z * z) - np.log(sigma)

        timed = uncen | cen
        lo, hi = truncation_bounds(x[timed])
        ll[timed] += weibull_logterms(x[timed, 1], uncen[timed].astype(float), lo, hi, kappa, lam)[0]

        assert not np.any(np.isnan(ll))
        return ll

    def estimator(self, x: np.ndarray, gammas: np.ndarray):
        """Weighted M step for a single condition."""
        fit_emission(self, x, gammas)

    def censor_lineage_array(
        self,
        censor_condition: int,
        tree: csr_array,
        obs: np.ndarray,
        states: np.ndarray,
        desired_experiment_time=2e12,
    ) -> tuple[csr_array, np.ndarray, np.ndarray]:
        """Applies censoring to array representation directly."""
        return censor_lineage_weibull(tree, obs, states, censor_condition, desired_experiment_time)
