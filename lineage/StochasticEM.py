"""Mini-batch (online) Baum-Welch for tHMMs, following the ideas of e-means.

Full-batch Baum-Welch recomputes every sufficient statistic from every lineage on
each iteration. Here, as in Cappé & Moulines' online EM and in e-means, each step runs
the E step on a mini-batch of lineages only and folds the result into running
sufficient statistics with an exponential moving average (EMA),

    S <- (1 - alpha_t) S + alpha_t * (N / |B|) * s_B,

with a step size alpha_t that is annealed by a cosine schedule. The noisy early steps
help the fit escape poor local optima; the annealed late steps settle it. States that
lose (almost) all of their occupancy early on are revived by splitting a well-occupied
state, and the run finishes with a few exact full-batch Baum-Welch iterations, which
remove the EMA lag and give an exact log-likelihood.

The transition matrix has finite additive sufficient statistics, so those are kept
directly. The emission estimators, however, take a per-cell weight vector (the
right-censored Gamma likelihood has no finite-dimensional sufficient statistic), so
for the emissions the EMA state *is* that weight vector: decaying every cell's weight
by (1 - alpha) and adding alpha * N / |B| times the fresh gammas of the batch cells is
exactly the EMA of sum_i w_i * phi(x_i) for any per-cell statistic phi. That keeps
every existing estimator (Bernoulli/Gamma, censored Gamma, competing risks, the
at-once multi-condition fits) exact and unchanged.
"""

from copy import deepcopy

import numpy as np

from .BaumWelch import (
    T_from_stats,
    T_sufficient_stats,
    calculate_log_likelihood,
    do_E_step,
    do_M_E_step,
    do_M_E_step_atonce,
    do_M_pi_E_step,
    do_M_step,
    sum_nonleaf_gammas,
)
from .tHMM import tHMM


def cosine_schedule(alpha0: float, n_steps: int, final_ratio: float = 0.1) -> np.ndarray:
    """Step sizes annealed from ``alpha0`` to ``final_ratio * alpha0`` along a half cosine."""
    if n_steps <= 1:
        return np.full(max(n_steps, 0), alpha0)
    alpha_min = final_ratio * alpha0
    t = np.arange(n_steps) / (n_steps - 1)
    return alpha_min + 0.5 * (alpha0 - alpha_min) * (1.0 + np.cos(np.pi * t))


def _make_batches(n_lineages: list[int], n_batches: int, rng) -> list[list[np.ndarray]]:
    """Shuffle each condition's lineages and split them into ``n_batches`` chunks.

    Batches are stratified: batch b holds the b-th chunk of every condition, so each
    step updates the emissions of every condition when fitting several at once.
    """
    per_cond = [np.array_split(rng.permutation(n), min(n_batches, n)) for n in n_lineages]
    nb = max(len(c) for c in per_cond)
    # Conditions with fewer lineages than batches simply sit out the surplus batches.
    return [[c[b] if b < len(c) else np.empty(0, dtype=int) for c in per_cond] for b in range(nb)]


def _M_step_from_stats(tHMMobj_list: list[tHMM], numer: np.ndarray, denom: np.ndarray, weights: list):
    """M step from the running statistics, mirroring :func:`~lineage.BaumWelch.do_M_step`."""
    if tHMMobj_list[0].estimate.fT is None:
        T = T_from_stats(numer, denom)
        for t in tHMMobj_list:
            t.estimate.T = T

    do_M_pi_E_step(tHMMobj_list, weights)


def _revive_states(
    tHMMobj_list: list[tHMM],
    weights: list[list[np.ndarray]],
    numer: np.ndarray,
    denom: np.ndarray,
    threshold: float,
    rng,
) -> int:
    """Split a well-occupied state into any state whose occupancy has collapsed.

    A state is starved when its total weight falls below ``threshold`` times the
    average per-state occupancy. A donor is drawn with probability proportional to
    occupancy (as in e-means). The donor's cells it explains worst (below-median
    emission log-likelihood) mostly move to the revived state, which gives the two
    copies different emission fits rather than two near-identical halves.

    :return: the number of states revived
    """
    occupancy = np.sum([np.vstack(w).sum(axis=0) for w in weights], axis=0)
    K = occupancy.size
    starved = np.nonzero(occupancy < threshold * occupancy.sum() / K)[0]
    alive = np.setdiff1d(np.arange(K), starved)
    if starved.size == 0 or alive.size == 0:
        return 0

    for dead in starved:
        p = occupancy[alive] / occupancy[alive].sum()
        donor = int(rng.choice(alive, p=p))

        for tO, w in zip(tHMMobj_list, weights, strict=True):
            cells = np.vstack([lin.obs for lin in tO.X])
            ll = tO.estimate.E[donor].logpdf(cells)
            share = np.where(ll < np.median(ll), 0.75, 0.25)
            share *= rng.uniform(0.9, 1.1, size=share.size)

            start = 0
            for lin_w in w:
                n = lin_w.shape[0]
                moved = lin_w[:, donor] * share[start : start + n]
                lin_w[:, dead] += moved
                lin_w[:, donor] -= moved
                start += n

        # Split the donor's transition statistics evenly: first its column (being
        # transitioned into), then its row (transitioning out of).
        numer[:, dead] += 0.5 * numer[:, donor]
        numer[:, donor] *= 0.5
        numer[dead, :] += 0.5 * numer[donor, :]
        numer[donor, :] *= 0.5
        denom[dead] += 0.5 * denom[donor]
        denom[donor] *= 0.5

        occupancy[dead] = occupancy[donor] = 0.5 * (occupancy[donor] + occupancy[dead])

    return int(starved.size)


def fit_list_stochastic(
    tHMMobj_list: list[tHMM],
    epochs: int = 10,
    n_batches: int = 4,
    alpha0: float | None = None,
    final_ratio: float = 0.1,
    revive_threshold: float = 0.05,
    polish_iter: int = 10,
    rng=None,
) -> tuple[list, list, float]:
    """
    Mini-batch Baum-Welch with EMA sufficient statistics (see the module docstring).

    Drop-in alternative to :func:`~lineage.Analyze.fit_list`, with the same inputs and
    outputs.

    :param tHMMobj_list: all lineage trees we want to fit at once (one tHMM per condition)
    :param epochs: passes over the data in the stochastic phase
    :param n_batches: mini-batches per epoch (per condition, stratified)
    :param alpha0: initial EMA step size; defaults to ``min(1, 4 / n_batches)``. With the
        default four batches the first step simply replaces the initial statistics, and
        by the end of the cosine schedule the running average spans about ten steps.
        e-means sets this from fitted scaling laws in (n, k, T); those do not transfer
        to tree HMMs, so this default was instead chosen by a sweep on the lab's
        lineage datasets.
    :param final_ratio: the schedule anneals alpha from ``alpha0`` to ``final_ratio * alpha0``
    :param revive_threshold: a state whose occupancy falls below this fraction of the
        average per-state occupancy during the first half of training is revived by
        splitting another state. Set to 0 to disable.
    :param polish_iter: number of exact full-batch Baum-Welch iterations at the end;
        the parameters with the best full-data log-likelihood seen are kept
    :return: (NF_list, gammas_list, LL), as for :func:`~lineage.Analyze.fit_list`
    """
    rng = np.random.default_rng(rng)
    K = tHMMobj_list[0].num_states
    n_lineages = [len(tO.X) for tO in tHMMobj_list]
    fit_T = tHMMobj_list[0].estimate.fT is None

    if alpha0 is None:
        alpha0 = min(1.0, 4.0 / n_batches)

    # Step 0: random responsibilities, as for full-batch fitting. These seed the
    # running per-cell weights, and an M step from them seeds the emissions.
    weights = [[rng.dirichlet(np.ones(K), size=len(lin)) for lin in tO.X] for tO in tHMMobj_list]
    if len(tHMMobj_list) > 1:
        do_M_E_step_atonce(tHMMobj_list, weights)
    else:
        do_M_E_step(tHMMobj_list[0], weights[0])

    # Seed the transition statistics so they reproduce the initial T, with the
    # occupancy the initial weights imply.
    denom = 1e-12 + np.sum(
        [
            sum_nonleaf_gammas(lin.leaves_idx, w)
            for tO, ws in zip(tHMMobj_list, weights, strict=True)
            for lin, w in zip(tO.X, ws, strict=True)
        ],
        axis=0,
    )
    numer = tHMMobj_list[0].estimate.T * denom[:, np.newaxis]

    n_steps = epochs * max(1, min(n_batches, max(n_lineages)))
    alphas = cosine_schedule(alpha0, n_steps, final_ratio)

    step = 0
    while step < n_steps:
        for batch in _make_batches(n_lineages, n_batches, rng):
            if step >= n_steps:
                break
            alpha = alphas[step]

            numer_B = np.zeros((K, K))
            denom_B = np.zeros(K)
            for c, (tO, idx) in enumerate(zip(tHMMobj_list, batch, strict=True)):
                if idx.size == 0:
                    continue
                lins = [tO.X[i] for i in idx]
                MSD, _, betas, gammas = do_E_step(tO, lins)
                scale = n_lineages[c] / idx.size

                # EMA of the per-cell weights for this condition
                for w in weights[c]:
                    w *= 1.0 - alpha
                for i, g in zip(idx, gammas, strict=True):
                    weights[c][i] += alpha * scale * g

                if fit_T:
                    nB, dB = T_sufficient_stats(lins, MSD, betas, gammas, tO.estimate.T)
                    numer_B += scale * nB
                    denom_B += scale * dB

            if fit_T:
                numer = (1.0 - alpha) * numer + alpha * numer_B
                denom = (1.0 - alpha) * denom + alpha * denom_B

            if revive_threshold > 0 and step < n_steps // 2:
                _revive_states(tHMMobj_list, weights, numer, denom, revive_threshold, rng)

            _M_step_from_stats(tHMMobj_list, numer, denom, weights)
            step += 1

    # Exact full-batch finalization. Baum-Welch with these estimators is not strictly
    # monotone, so keep whichever iterate has the best full-data log-likelihood.
    best = None
    for it in range(polish_iter + 1):
        MSD_list, NF_list, betas_list, gammas_list = map(
            list, zip(*[do_E_step(tO) for tO in tHMMobj_list], strict=False)
        )
        LL = calculate_log_likelihood(NF_list)
        if best is None or LL > best[0]:
            best = (LL, NF_list, gammas_list, [deepcopy(tO.estimate) for tO in tHMMobj_list])
        if it < polish_iter:
            do_M_step(tHMMobj_list, MSD_list, betas_list, gammas_list)

    assert best is not None
    LL, NF_list, gammas_list, estimates = best
    for tO, est in zip(tHMMobj_list, estimates, strict=True):
        tO.estimate = est

    return NF_list, gammas_list, LL
