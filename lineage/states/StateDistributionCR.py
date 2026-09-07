"""Competing-risks state distributions.

The Gamma/GaPhs emissions in this package treat a cell's fate (Bernoulli) and its
phase duration (Gamma) as independent observations, and discard the duration of any
cell that dies.  That throws away every death time and, more subtly, mis-states the
likelihood of a time-censored cell: "no event yet at time t" is written as
P(division > t) when it should be P(division > t AND death > t).

Here each phase instead carries two latent clocks,

    T_D ~ Gamma(a, s)          division / transition
    T_X ~ Gamma(a_x, s_x)      death

and we observe min(T_D, T_X) together with an indicator of which fired.  The three
likelihood cases are the standard competing-risks ones:

    transition seen at t    f_D(t) * S_X(t)
    death seen at t         f_X(t) * S_D(t)
    censored at t           S_D(t) * S_X(t)

The division probability is then *derived*, P(divide) = int f_D(t) S_X(t) dt, rather
than fit as a free Bernoulli parameter, so the death fraction and the death timing are
forced to agree.  With the G1 death clock pinned to a constant hazard (shape 1) this
costs no degrees of freedom relative to the Bernoulli/Gamma model.
"""

from typing import Literal

import numpy as np
import scipy.stats as sp
from scipy.integrate import quad
from scipy.sparse import csr_array

from .stateCommon import censor_lineage_gamma, censor_lineage_gaphs, gamma_estimator

# Smallest duration fed to a pdf, so that a recorded duration of exactly zero cannot
# produce a non-finite emission likelihood.
TIME_FLOOR = 1e-10


def event_masks(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split cells into the three competing-risks cases.

    ``x`` has the usual three per-phase columns ``[fate, duration, censoring]``, where
    fate is 1 for surviving the phase, 0 for dying in it, and NaN when unknown, and the
    censoring flag is 1 when the phase was seen through to its end.

    Cells whose duration is negative have been masked for cross validation, and cells
    with a NaN duration never entered the phase; both are excluded everywhere.

    :return: boolean masks for (division observed, death observed, censored)
    """
    fate, dur, cens = x[:, 0], x[:, 1], x[:, 2]

    valid = np.isfinite(dur) & (dur >= 0.0)
    died = valid & (fate == 0.0)
    divided = valid & (fate == 1.0) & (cens == 1.0)
    # Everything else that we timed is censored: an unknown fate, or a cell whose phase
    # we only caught part of (the root cell's G1, or a cell first seen in G2). Those
    # partial durations are lower bounds on the true one, so right-censoring them is
    # the conservative reading, and it is what the Bernoulli/Gamma model already did.
    censored = valid & ~died & ~divided

    return divided, died, censored


def exponential_estimator(obs: np.ndarray, events: np.ndarray, weights: np.ndarray, param_idx: np.ndarray, K: int):
    """Weighted right-censored MLE for exponential scales, one per group.

    With a constant hazard the MLE is total time at risk over events observed, which
    needs no iteration.  A pseudocount keeps a group with no observed deaths finite.
    """
    scales = np.empty(K)
    for k in range(K):
        sel = param_idx == (k + 1)
        at_risk = float(np.dot(weights[sel], obs[sel])) + 1.0
        n_events = float(np.dot(weights[sel], events[sel])) + 1.0 / K
        scales[k] = at_risk / n_events
    return scales


def fit_clocks(distributions, x_list: list[np.ndarray], gammas_list: list[np.ndarray], state_j: int):
    """Fit both clocks of one state, sharing each shape across conditions.

    ``distributions`` is either a single :class:`StateDistribution` or a list of them,
    one per condition.  The shape parameters are shared across conditions and the
    scales are free, mirroring how ``atonce_estimator`` treats the Gamma model.
    """
    single = not isinstance(distributions, list)
    dists = [distributions] if single else distributions
    K = len(x_list)

    x = np.concatenate(x_list, axis=0)
    weights = np.concatenate([g[:, state_j] for g in gammas_list])
    idx = np.concatenate([np.full(g.shape[0], k + 1) for k, g in enumerate(gammas_list)])

    divided, died, censored = event_masks(x)
    timed = divided | died | censored
    if not np.any(timed):
        return

    t = np.clip(x[timed, 1], TIME_FLOOR, None)
    w = weights[timed]
    idx = idx[timed]
    div_event = divided[timed].astype(float)
    death_event = died[timed].astype(float)

    ref = dists[0]

    # Division clock: every timed cell contributes, as an event or as censored.
    if np.any(div_event > 0.0):
        x0 = np.array([ref.params[1]] + [d.params[2] for d in dists])
        out = gamma_estimator(t, div_event, w, idx, x0, phase="all")
        for k, d in enumerate(dists):
            d.params[1] = out[0]
            d.params[2] = out[k + 1]

    # Death clock: same cells, with the roles of event and censoring swapped.
    if np.any(death_event > 0.0):
        if ref.fixed_death_shape:
            scales = exponential_estimator(t, death_event, w, idx, K)
            for k, d in enumerate(dists):
                d.params[3] = 1.0
                d.params[4] = scales[k]
        else:
            x0 = np.array([ref.params[3]] + [d.params[4] for d in dists])
            out = gamma_estimator(t, death_event, w, idx, x0, phase="all")
            for k, d in enumerate(dists):
                d.params[3] = out[0]
                d.params[4] = out[k + 1]

    for d in dists:
        d.params[0] = d.division_probability()


def atonce_estimator(
    all_tHMMobj: list,
    x_list: list,
    gammas_list: list[np.ndarray],
    phase: Literal["all", "G1", "G2"],
):
    """M step across several conditions at once, matching the Gamma model's interface."""
    x_list = [np.asarray(x) for x in x_list]

    for state_j in range(len(all_tHMMobj[0].estimate.E)):
        emissions = [tO.estimate.E[state_j] for tO in all_tHMMobj]

        if phase == "all":
            fit_clocks(emissions, x_list, gammas_list, state_j)
        else:
            sub = "G1" if phase == "G1" else "G2"
            fit_clocks([getattr(e, sub) for e in emissions], x_list, gammas_list, state_j)
            for e in emissions:
                e._sync()


class StateDistribution:
    """One cell-cycle phase with competing division and death clocks.

    ``params`` is ``[bern_p, gamma_a, gamma_scale, death_a, death_scale]``. The first
    three entries keep the meaning they have in
    :class:`~lineage.states.StateDistributionGamma.StateDistribution`, so downstream
    figure code that indexes them positionally continues to work; ``bern_p`` is now
    derived from the two clocks rather than fit.
    """

    #: BaumWelch looks this up on the emission object to pick the right M step.
    atonce_estimator = staticmethod(atonce_estimator)

    def __init__(
        self,
        gamma_a: float = 7.0,
        gamma_scale: float = 4.5,
        death_a: float = 1.0,
        death_scale: float = 40.0,
        fixed_death_shape: bool = True,
    ):
        """
        :param fixed_death_shape: pin the death clock's shape at 1, i.e. a constant
            death hazard. True for G1, where the death times are memoryless; False for
            G2, where they show a strongly increasing hazard.
        """
        self.fixed_death_shape = fixed_death_shape
        if fixed_death_shape:
            death_a = 1.0
        self.params = np.array([0.0, gamma_a, gamma_scale, death_a, death_scale])
        self.params[0] = self.division_probability()

    # -- the two clocks ------------------------------------------------------

    @property
    def div_clock(self):
        return sp.gamma(a=self.params[1], scale=self.params[2])

    @property
    def death_clock(self):
        return sp.gamma(a=self.params[3], scale=self.params[4])

    def division_probability(self) -> float:
        """P(the division clock fires first) = int f_D(t) S_X(t) dt."""
        div, death = self.div_clock, self.death_clock

        def integrand(t):
            return div.pdf(t) * death.sf(t)

        upper = div.ppf(1.0 - 1e-9)
        val = quad(integrand, 0.0, upper, limit=100)[0]
        return float(np.clip(val, 0.0, 1.0))

    def rvs(self, size: int, rng=None):
        """Draw min(T_D, T_X) and record which clock fired."""
        rng = np.random.default_rng(rng)
        t_div = rng.gamma(self.params[1], scale=self.params[2], size=size)
        t_death = rng.gamma(self.params[3], scale=self.params[4], size=size)

        divided = t_div <= t_death
        return divided.astype(float), np.minimum(t_div, t_death), np.ones(size)

    def dist(self, other) -> float:
        """Wasserstein distance between the division clocks of two states.

        Kept on the division clock alone so that the number is comparable with the
        Bernoulli/Gamma model's.
        """
        assert isinstance(self, type(other))
        return float(np.absolute(self.params[1] * self.params[2] - other.params[1] * other.params[2]))

    def dof(self) -> int:
        """Two division-clock parameters plus the death clock's scale, and its shape
        when that is not pinned. The Bernoulli is derived, so it is not counted."""
        return 3 if self.fixed_death_shape else 4

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """Competing-risks log likelihood of each cell's phase observation."""
        divided, died, censored = event_masks(x)
        timed = divided | died | censored
        t = np.clip(x[:, 1], TIME_FLOOR, None)

        div, death = self.div_clock, self.death_clock

        ll = np.zeros(x.shape[0])
        # Every timed cell survived both clocks up to t; the one that fired then
        # swaps its survival term for a density.
        ll[timed] += div.logsf(t[timed]) + death.logsf(t[timed])
        ll[divided] += div.logpdf(t[divided]) - div.logsf(t[divided])
        ll[died] += death.logpdf(t[died]) - death.logsf(t[died])

        assert not np.any(np.isnan(ll))
        return ll

    def estimator(self, x: np.ndarray, gammas: np.ndarray):
        """Weighted M step for a single condition.

        The two clocks separate in the complete-data likelihood, so each is just a
        weighted right-censored Gamma fit over every timed cell.
        """
        fit_clocks(self, [x], [gammas[:, np.newaxis]], state_j=0)

    def censor_lineage_array(
        self,
        censor_condition: int,
        tree: csr_array,
        obs: np.ndarray,
        states: np.ndarray,
        desired_experiment_time=2e12,
    ) -> tuple[csr_array, np.ndarray, np.ndarray]:
        """Applies censoring to array representation directly."""
        return censor_lineage_gamma(tree, obs, states, censor_condition, desired_experiment_time)


class StateDistributionPhase:
    """G1 and G2 phases, each with its own pair of competing clocks.

    ``params`` is ``[bern_p1, bern_p2, a1, s1, a2, s2, death_a1, death_s1, death_a2,
    death_s2]``. The leading six entries match
    :class:`~lineage.states.StateDistributionGaPhs.StateDistribution` exactly.
    """

    #: BaumWelch looks this up on the emission object to pick the right M step.
    atonce_estimator = staticmethod(atonce_estimator)

    def __init__(
        self,
        gamma_a1: float = 7.0,
        gamma_scale1: float = 3.0,
        gamma_a2: float = 14.0,
        gamma_scale2: float = 6.0,
        death_scale1: float = 40.0,
        death_a2: float = 3.0,
        death_scale2: float = 20.0,
    ):
        # G1 deaths are memoryless, so that clock is a one-parameter exponential; G2
        # deaths have a strongly increasing hazard and need a free shape.
        self.G1 = StateDistribution(gamma_a1, gamma_scale1, 1.0, death_scale1, fixed_death_shape=True)
        self.G2 = StateDistribution(gamma_a2, gamma_scale2, death_a2, death_scale2, fixed_death_shape=False)
        self.params = np.empty(10)
        self._sync()

    def _sync(self):
        """Mirror the sub-distributions' parameters into the flat ``params`` array."""
        self.params[0] = self.G1.params[0]
        self.params[1] = self.G2.params[0]
        self.params[2:4] = self.G1.params[1:3]
        self.params[4:6] = self.G2.params[1:3]
        self.params[6:8] = self.G1.params[3:5]
        self.params[8:10] = self.G2.params[3:5]

    def rvs(self, size: int, rng=None):
        rng = np.random.default_rng(rng)
        bern_G1, gamma_G1, cens_G1 = self.G1.rvs(size, rng=rng)
        bern_G2, gamma_G2, cens_G2 = self.G2.rvs(size, rng=rng)
        return bern_G1, bern_G2, gamma_G1, gamma_G2, cens_G1, cens_G2

    def dist(self, other) -> float:
        assert isinstance(self, type(other))
        return self.G1.dist(other.G1) + self.G2.dist(other.G2)

    def dof(self) -> int:
        return self.G1.dof() + self.G2.dof()

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return self.G1.logpdf(x[:, np.array([0, 2, 4])]) + self.G2.logpdf(x[:, np.array([1, 3, 5])])

    def estimator(self, x: np.ndarray, gammas: np.ndarray):
        self.G1.estimator(x[:, np.array([0, 2, 4])], gammas)
        self.G2.estimator(x[:, np.array([1, 3, 5])], gammas)
        self._sync()

    def censor_lineage_array(
        self,
        censor_condition: int,
        tree: csr_array,
        obs: np.ndarray,
        states: np.ndarray,
        desired_experiment_time=2e12,
    ) -> tuple[csr_array, np.ndarray, np.ndarray]:
        return censor_lineage_gaphs(tree, obs, states, censor_condition, desired_experiment_time)
