"""Heritability of hidden states: persistence, half-lives, and dose-dependent transitions.

For a two-state model with state 0 the drug-sensitive (arrested) state and state 1 the
drug-tolerant (cycling) state, the diagonal :math:`T_{kk}` is the probability that a
daughter keeps its mother's state. Along a single line of descent the chance of staying
in state ``k`` for ``g`` consecutive divisions is :math:`T_{kk}^g`, so the persistence
half-life is

.. math:: \\tau_{1/2} = \\ln(1/2) / \\ln T_{kk} \\quad\\text{(generations)}.

:math:`T_{kk}` alone does not say whether the state is heritable, though: a state that is
redrawn at random every division has :math:`T_{kk} = \\pi_k`, its stationary frequency,
which can be anything. Heritability is the dependence of the daughter's state on the
mother's, i.e. how far the rows of ``T`` differ. For ``T`` with a stationary
distribution this is summarised by its subdominant eigenvalue :math:`\\lambda_2`
(:math:`T_{11} + T_{22} - 1` when there are two states): the correlation between the
states of cells ``g`` generations apart decays as :math:`\\lambda_2^g`, and it is exactly
zero -- no memory -- when every row of ``T`` is the same. :func:`dose_sweep` tests that
null directly with a likelihood-ratio test.
"""

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import chi2

from .Analyze import fit_list
from .BaumWelch import do_E_step
from .states.CensoredWeibullGaussian import StateDistribution
from .tHMM import tHMM


def persistence_half_life(T: np.ndarray, state: int) -> float:
    """Generations over which a line of descent stays in ``state`` with probability 1/2."""
    p = float(T[state, state])
    if p >= 1.0:
        return np.inf
    if p <= 0.0:
        return 0.0
    return float(np.log(0.5) / np.log(p))


def memory_eigenvalue(T: np.ndarray) -> float:
    """The subdominant eigenvalue of ``T``, which sets how fast state memory decays.

    For two states this is :math:`T_{00} + T_{11} - 1`. It is 0 when daughters' states
    are independent of their mother's, 1 when states never switch, and negative when
    daughters tend to take the opposite state to their mother.
    """
    ev = np.linalg.eigvals(T)
    ev = ev[np.argsort(-np.abs(ev))]
    # The leading eigenvalue of a stochastic matrix is exactly 1.
    return float(np.real(ev[1]))


def memory_half_life(T: np.ndarray) -> float:
    """Generations for the mother-descendant state correlation to halve, :math:`\\ln(1/2)/\\ln|\\lambda_2|`."""
    lam = abs(memory_eigenvalue(T))
    if lam >= 1.0:
        return np.inf
    if lam <= 1e-12:  # no memory, up to rounding in the eigensolver
        return 0.0
    return float(np.log(0.5) / np.log(lam))


def order_states(tHMMobj: tHMM, order: np.ndarray):
    """Relabel the states of a fitted model in place so that new state ``i`` is old state ``order[i]``."""
    est = tHMMobj.estimate
    est.T = est.T[np.ix_(order, order)]
    est.pi = est.pi[order]
    est.E = [est.E[i] for i in order]


def order_by_lifetime(tHMMobj_list: list[tHMM]) -> np.ndarray:
    """Put the slowest-dividing state first (the arrested one) and the fastest last.

    With shared emissions the order is the same for every condition, so it is taken from
    the first model and applied to all of them.
    """
    E = tHMMobj_list[0].estimate.E
    assert all(isinstance(e, StateDistribution) for e in E)
    order = np.argsort([-e.mean_lifetime() for e in E if isinstance(e, StateDistribution)])
    for tO in tHMMobj_list:
        order_states(tO, order)
    return order


def refit(
    pop_list: list,
    template: list[tHMM],
    shared_T: bool,
    independent_T: bool = False,
    estimate_pi: bool = True,
    rng=None,
) -> tuple[list[tHMM], float, list]:
    """Run EM from the parameters of an already-fit model (e.g. for a bootstrap replicate
    or to start a larger model from a nested one)."""
    objs = []
    for X, tmpl in zip(pop_list, template, strict=True):
        tO = tHMM(X, num_states=tmpl.num_states, rng=rng)
        tO.estimate.T = tmpl.estimate.T.copy()
        tO.estimate.pi = tmpl.estimate.pi.copy()
        tO.estimate.E = deepcopy(tmpl.estimate.E)
        objs.append(tO)
    _, gammas, LL = fit_list(
        objs, rng=rng, shared_T=shared_T, independent_T=independent_T, random_init=False, estimate_pi=estimate_pi
    )
    return objs, LL, gammas


def _random_fit(pop_list, num_states, shared_T, independent_T, estimate_pi, seed):
    """One EM run from a random start; top level so that it can run in a worker process."""
    rng = np.random.default_rng(seed)
    objs = [tHMM(X, num_states=num_states, rng=rng) for X in pop_list]
    _, _, LL = fit_list(objs, rng=rng, shared_T=shared_T, independent_T=independent_T, estimate_pi=estimate_pi)
    return objs, LL


def fit_best(
    pop_list: list,
    num_states: int,
    shared_T: bool,
    independent_T: bool = False,
    starts=(),
    n_starts: int = 6,
    estimate_pi: bool = True,
    n_jobs: int = 1,
    rng=None,
) -> tuple[list[tHMM], float]:
    """Best of ``n_starts`` random restarts and of warm starts from each model in ``starts``.

    Root-state distributions are estimated freely by default (``estimate_pi``): lineage
    roots are generally not drawn from the stationary distribution of ``T`` -- in a drug
    time course they are cells from before the drug -- and the stationary tie also
    breaks the monotonicity of EM, which the likelihood-ratio tests rely on.

    :param n_jobs: run the random restarts in this many processes
    """
    rng = np.random.default_rng(rng)
    seeds = rng.integers(2**32, size=n_starts)
    args = (pop_list, num_states, shared_T, independent_T, estimate_pi)
    if n_jobs > 1:
        with ProcessPoolExecutor(min(n_jobs, n_starts)) as exe:
            fits = list(exe.map(_random_fit, *zip(*[(*args, s) for s in seeds], strict=True)))
    else:
        fits = [_random_fit(*args, s) for s in seeds]
    for s in starts:
        objs, LL, _ = refit(pop_list, s, shared_T, independent_T, estimate_pi, rng=rng)
        fits.append((objs, LL))
    return max(fits, key=lambda f: f[1])


@dataclass
class DoseSweep:
    """Result of :func:`dose_sweep`. States are ordered slowest-dividing first."""

    doses: list
    per_dose: list[tHMM]
    shared: list[tHMM]
    independent: list[tHMM]
    LL: dict[str, float]
    lrt: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def T(self) -> np.ndarray:
        """Per-dose transition matrices, shape (doses, K, K)."""
        return np.stack([tO.estimate.T for tO in self.per_dose])

    def half_lives(self, state: int = -1) -> np.ndarray:
        """Per-dose persistence half-life of ``state`` (default: the fastest-cycling one)."""
        state = state % self.T.shape[1]
        return np.array([persistence_half_life(T, state) for T in self.T])

    def memory(self) -> np.ndarray:
        """Per-dose subdominant eigenvalue of T."""
        return np.array([memory_eigenvalue(T) for T in self.T])


def lrt(LL_null: float, LL_alt: float, df: int) -> dict[str, float]:
    """Likelihood-ratio test. The statistic is clipped at zero, since EM noise can leave a
    nested alternative a hair below its null."""
    stat = max(2.0 * (LL_alt - LL_null), 0.0)
    return {"statistic": stat, "df": df, "p": float(chi2.sf(stat, df))}


def dose_sweep(
    pops_by_dose: list[list], doses: list, num_states: int = 2, n_starts: int = 6, n_jobs: int = 1, rng=None
) -> DoseSweep:
    """Fit a tHMM across doses with shared emissions under three transition models.

    * ``per_dose`` -- a separate transition matrix for every dose;
    * ``shared`` -- one transition matrix for all doses (does the transition matrix depend on dose?);
    * ``independent`` -- a separate matrix per dose, but with identical rows, so that a
      daughter's state is independent of its mother's (is the state heritable at all?).

    Emissions are shared across doses so that a state names the same phenotype at every
    dose, and each dose has its own root-state distribution. The three models are
    cross-seeded -- each is also started from the others' solutions -- so that the nested
    comparisons are not decided by which random restart happened to find a better optimum.
    """
    rng = np.random.default_rng(rng)
    K, D = num_states, len(pops_by_dose)
    kw: dict = {"n_starts": n_starts, "n_jobs": n_jobs, "rng": rng}

    shared = fit_best(pops_by_dose, K, shared_T=True, **kw)
    indep = fit_best(pops_by_dose, K, shared_T=False, independent_T=True, **kw)
    per_dose = fit_best(pops_by_dose, K, shared_T=False, starts=(shared[0], indep[0]), **kw)
    shared = fit_best(pops_by_dose, K, shared_T=True, starts=(shared[0], per_dose[0]), n_starts=0, rng=rng)
    indep = fit_best(
        pops_by_dose, K, shared_T=False, independent_T=True, starts=(indep[0], per_dose[0]), n_starts=0, rng=rng
    )
    per_dose = fit_best(pops_by_dose, K, shared_T=False, starts=(per_dose[0], shared[0], indep[0]), n_starts=0, rng=rng)

    for fit in (per_dose, shared, indep):
        order_by_lifetime(fit[0])

    LL = {"per_dose": per_dose[1], "shared": shared[1], "independent": indep[1]}
    tests = {
        "dose_dependence": lrt(shared[1], per_dose[1], (D - 1) * K * (K - 1)),
        "heritability": lrt(indep[1], per_dose[1], D * (K - 1) ** 2),
    }
    return DoseSweep(list(doses), per_dose[0], shared[0], indep[0], LL, tests)


def bootstrap_transitions(sweep: DoseSweep, pops_by_dose: list[list], n_boot: int = 100, rng=None) -> np.ndarray:
    """Nonparametric bootstrap of the per-dose transition matrices.

    Whole lineages are resampled with replacement within each dose (cells in a lineage
    are not independent), and each replicate is refit from the full-data estimate.

    :return: array of shape (n_boot, doses, K, K)
    """
    rng = np.random.default_rng(rng)
    out = []
    for _ in range(n_boot):
        boot = [[pop[i] for i in rng.integers(len(pop), size=len(pop))] for pop in pops_by_dose]
        objs, _, _ = refit(boot, sweep.per_dose, shared_T=False, rng=rng)
        # A replicate can converge with its labels swapped; restore the lifetime ordering.
        order_by_lifetime(objs)
        out.append(np.stack([tO.estimate.T for tO in objs]))
    return np.stack(out)


def commitment_trees(tHMMobj: tHMM) -> list[dict[str, np.ndarray]]:
    """Exact posterior of each mother-daughter pair's joint state, lineage by lineage.

    :return: one dict per lineage with the ``parents`` and ``daughters`` edge indices,
        ``pair`` (edges, K, K), the posterior :math:`P(z_p = k, z_d = l \\mid X)`, and
        ``p_switch``, the posterior probability that the daughter left its mother's state
    """
    MSD, _, betas, gammas = do_E_step(tHMMobj)
    T = tHMMobj.estimate.T
    eps = np.finfo(float).eps
    out = []
    for lO, msd, beta, gamma in zip(tHMMobj.X, MSD, betas, gammas, strict=True):
        parents, daughters = lO.edges
        betaMSD = beta / np.clip(msd, eps, None)
        TbetaMSD = np.clip(betaMSD @ T.T, eps, None)
        # Same factorisation as lineage.HMM.M_step.get_all_zetas, but kept per edge.
        pair = (gamma[parents] / TbetaMSD[daughters])[:, :, None] * T[None] * betaMSD[daughters][:, None, :]
        out.append(
            {
                "parents": parents,
                "daughters": daughters,
                "pair": pair,
                "p_switch": 1.0 - np.einsum("ekk->e", pair),
            }
        )
    return out
