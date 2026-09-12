"""Cross-validated comparison of the Gamma/Bernoulli and competing-risks emissions.

The two emissions are densities over *different* observation spaces. The competing-risks
form in :mod:`lineage.states.StateDistributionCR` puts a density on when a cell died,
which the Gamma/Bernoulli form has no way to express -- it scores a death as a bare
Bernoulli outcome and discards the time. Comparing their raw likelihoods would charge
the competing-risks model for predicting strictly more.

The headline metric therefore coarsens death timing away, leaving a space both models
describe:

    ============  =========================  ====================
    outcome       competing risks            Gamma/Bernoulli
    ============  =========================  ====================
    transition t  ``f_D(t) S_X(t)``          ``p f_D(t)``
    death         ``1 - P(divide)``          ``1 - p``
    censored, c   ``S_D(c) S_X(c)``          ``S_D(c)``
    ============  =========================  ====================

The transition branches carry identical total mass, so that comparison is like for
like. The censored branch is where the Gamma/Bernoulli form ignores that a censored
cell also did not die, and so claims more probability than it is entitled to -- see
:func:`outcome_mass`, which shows its total exceeding one by up to ~10% for the
short-horizon cells that make up much of this data. That bias runs in the Gamma
model's favour, which makes the comparison conservative.

Run as ``python -m lineage.compare_emissions <population> <states> <reps>``, e.g.
``python -m lineage.compare_emissions AllLapatinib 2,3,4 5``.
"""

import json
import sys
import time

import numpy as np
import scipy.stats as sp
from scipy.special import logsumexp

from .Analyze import Analyze_list
from .LineageTree import LineageTree
from .states.StateDistributionCR import StateDistributionPhase as CR
from .states.StateDistributionCR import event_masks
from .states.StateDistributionGaPhs import StateDistribution as GA

MODELS = {"gamma": GA, "cr": CR}

#: Column triples ``[fate, duration, censoring]`` for G1 and G2 in the phase observation.
PHASE_COLS = (np.array([0, 2, 4]), np.array([1, 3, 5]))

TIME_FLOOR = 1e-10


def outcome_mass(a: float, scale: float, p_div: float, horizon: float) -> tuple[float, float]:
    """Total probability each emission assigns across the outcomes of one phase.

    For a cell watched until ``horizon`` the outcomes are: transition at some
    ``t <= horizon``, death at some ``t <= horizon``, or still going at the horizon. A
    well-formed emission spreads exactly probability one over those.

    :return: (Gamma/Bernoulli mass, competing-risks mass)
    """
    death_scale = a * scale / max(1.0 - p_div, 1e-12)
    div = sp.gamma(a, scale=scale)
    death = sp.expon(scale=death_scale)

    gamma_mass = p_div * div.cdf(horizon) + (1.0 - p_div) + div.sf(horizon)

    t = np.linspace(TIME_FLOOR, horizon, 200001)
    cr_mass = (
        np.trapezoid(div.pdf(t) * death.sf(t), t)
        + np.trapezoid(death.pdf(t) * div.sf(t), t)
        + div.sf(horizon) * death.sf(horizon)
    )
    return float(gamma_mass), float(cr_mass)


def coarse_logpdf(dist, x: np.ndarray) -> np.ndarray:
    """Log likelihood of a two-phase observation with death timing coarsened away."""
    if isinstance(dist, GA):
        # The Gamma/Bernoulli emission already coarsens: a death contributes log(1 - p).
        return dist.logpdf(x)

    out = np.zeros(x.shape[0])
    for cols, sub in zip(PHASE_COLS, (dist.G1, dist.G2), strict=True):
        xp = x[:, cols]
        divided, died, censored = event_masks(xp)
        t = np.clip(xp[:, 1], TIME_FLOOR, None)
        div, death = sub.div_clock, sub.death_clock

        survived = divided | censored
        out[survived] += div.logsf(t[survived]) + death.logsf(t[survived])
        out[divided] += div.logpdf(t[divided]) - div.logsf(t[divided])
        out[died] += np.log(max(1.0 - sub.params[0], 1e-300))
    return out


def build(pops: list, cls, num_states: int, mask_seed=None, frac: float = 0.25):
    """Rebuild populations under emission class ``cls``, masking ``frac`` of cells.

    Masking is driven by an rng over the tree shapes alone, so a given ``mask_seed``
    hides exactly the same cells whichever emission class is used, and the two models
    are scored on identical held-out sets.

    :return: (populations, held-out records of ``(lineage index, cell indices, obs)``)
    """
    E = [cls() for _ in range(num_states)]
    out, held = [], []
    rng = np.random.default_rng(mask_seed) if mask_seed is not None else None

    for pop in pops:
        trees, hidden = [], []
        for li, lin in enumerate(pop):
            obs = lin.obs.copy()
            if rng is not None:
                m = rng.random(obs.shape[0]) < frac
                hidden.append((li, np.nonzero(m)[0], obs[m].copy()))
                # Negating an observation is how this package marks it hidden.
                obs[m] *= -1.0
            trees.append(LineageTree(lin.tree, E, obs=obs, states=lin.states))
        out.append(trees)
        held.append(hidden)
    return out, held


def _state_weights(gammas, ci: int, li: int, idxs: np.ndarray) -> np.ndarray:
    """Normalized posterior over states for the given held-out cells."""
    w = np.clip(gammas[ci][li][idxs, :], 1e-300, None)
    return w / w.sum(axis=1, keepdims=True)


def heldout_LL(objs: list, gammas: list, held: list, scorer=coarse_logpdf) -> tuple[float, int]:
    """Held-out log likelihood, marginalized over the fitted state posterior."""
    tot, n = 0.0, 0
    for ci, tO in enumerate(objs):
        for li, idxs, true_obs in held[ci]:
            if len(idxs) == 0:
                continue
            lp = np.stack([scorer(tO.estimate.E[s], true_obs) for s in range(tO.num_states)], axis=1)
            tot += float(np.sum(logsumexp(lp + np.log(_state_weights(gammas, ci, li, idxs)), axis=1)))
            n += len(idxs)
    return tot, n


def fate_logloss(objs: list, gammas: list, held: list) -> tuple[float, int]:
    """Held-out log likelihood of the binary fate alone.

    Both models emit a division probability per phase, so this compares them over an
    identical observation space with no duration density involved at all.
    """
    tot, n = 0.0, 0
    for ci, tO in enumerate(objs):
        for li, idxs, true_obs in held[ci]:
            if len(idxs) == 0:
                continue
            w = _state_weights(gammas, ci, li, idxs)
            for ph in (0, 1):
                fate = true_obs[:, ph]
                known = np.isin(fate, (0.0, 1.0))
                if not np.any(known):
                    continue
                p = np.clip([tO.estimate.E[s].params[ph] for s in range(tO.num_states)], 1e-9, 1 - 1e-9)
                pf = np.where(fate[known, None] == 1.0, p[None, :], 1.0 - p[None, :])
                tot += float(np.sum(np.log(np.sum(w[known] * pf, axis=1))))
                n += int(known.sum())
    return tot, n


def death_time_LL(objs: list, gammas: list, held: list) -> tuple[float, int]:
    """Density the competing-risks model puts on held-out death times, given a death.

    This is the piece the Gamma/Bernoulli emission cannot score at all, so it is
    reported on its own rather than folded into the comparison. A positive value means
    the fitted death clock is sharper than a one-per-hour reference.
    """
    tot, n = 0.0, 0
    for ci, tO in enumerate(objs):
        for li, idxs, true_obs in held[ci]:
            if len(idxs) == 0:
                continue
            w = _state_weights(gammas, ci, li, idxs)
            for cols, attr in zip(PHASE_COLS, ("G1", "G2"), strict=True):
                xp = true_obs[:, cols]
                _, died, _ = event_masks(xp)
                if not np.any(died):
                    continue
                t = np.clip(xp[died, 1], TIME_FLOOR, None)
                lp = []
                for s in range(tO.num_states):
                    sub = getattr(tO.estimate.E[s], attr)
                    # Density of the death time conditional on death being the outcome.
                    lp.append(
                        sub.death_clock.logpdf(t) + sub.div_clock.logsf(t) - np.log(max(1.0 - sub.params[0], 1e-300))
                    )
                tot += float(np.sum(logsumexp(np.stack(lp, axis=1) + np.log(w[died]), axis=1)))
                n += int(died.sum())
    return tot, n


def run(pop_name: str, k_list: list[int], reps: int, seed0: int = 0):
    """Fit both emissions on the same masked data and emit one JSON record per rep."""
    from . import Lineage_collections as LC

    pops = getattr(LC, pop_name)

    for k in k_list:
        for r in range(reps):
            mask_seed = seed0 + 1000 * k + r
            row: dict = {"pop": pop_name, "k": k, "rep": r}

            for mname, cls in MODELS.items():
                trees, held = build(pops, cls, k, mask_seed=mask_seed)
                t0 = time.time()
                objs, LL, gam = Analyze_list(trees, k, rng=np.random.default_rng(mask_seed))

                coarse, n = heldout_LL(objs, gam, held)
                fate, n_fate = fate_logloss(objs, gam, held)
                entry = {
                    "trainLL": LL,
                    "coarse_heldout": coarse,
                    "n_heldout": n,
                    "fate_LL": fate,
                    "n_fate": n_fate,
                    "dof": objs[0].estimate.E[0].dof(),
                    "secs": time.time() - t0,
                    "params": [e.params.tolist() for e in objs[0].estimate.E],
                }
                if cls is CR:
                    entry["death_time_LL"], entry["n_deaths"] = death_time_LL(objs, gam, held)
                row[mname] = entry

            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    run(sys.argv[1], [int(v) for v in sys.argv[2].split(",")], int(sys.argv[3]))
