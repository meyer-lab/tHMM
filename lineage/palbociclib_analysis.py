"""Issue #1016: heritability of palbociclib escape in Spencer-lab MCF10A lineages.

Run as ``OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m lineage.palbociclib_analysis
[n_boot]``. The fits run in parallel worker processes, so BLAS's own threads should be
pinned to one, or the workers oversubscribe the machine. Results are written to
``output/palbociclib_analysis.json`` and drawn by :mod:`lineage.figures.figure22`.

See :mod:`lineage.palbociclib_loader` for the data, and why it is not the data set named
in the issue.
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .early_biomarker import FEATURES, auc_by_dose, auc_table, cross_validated_scores
from .heritability import (
    bootstrap_transitions,
    dose_sweep,
    fit_best,
    memory_eigenvalue,
    memory_half_life,
    order_by_lifetime,
    persistence_half_life,
)
from .palbociclib_loader import load_lineages
from .states.CensoredWeibullGaussian import StateDistribution

CONDITIONS = ("control", "palbociclib")
DOSES_NM = (0.0, 1000.0)
#: Escape = the daughter's CDK2 activity reaches S-phase levels (or it divides) within
#: this many hours of birth. The movie only runs ~22.6 h past drug addition, so the 48 h
#: lead time proposed in the issue cannot be scored on this data.
HORIZON_H = 12.0
#: Observation column holding each cell's S-phase entry time (see ``cell_obs``).
ESCAPE_COL = 3
OUTPUT = os.path.join("output", "palbociclib_analysis.json")


def n_observed(pops) -> int:
    return int(sum(np.sum(np.isfinite(lin.obs[:, :2]).any(axis=1)) for pop in pops for lin in pop))


def select_states(pops, max_states: int = 4, n_jobs: int = 1, rng=None) -> list[dict]:
    """BIC for 1..max_states states, with shared emissions and per-dose transitions and
    root-state distributions."""
    rng = np.random.default_rng(rng)
    n = n_observed(pops)
    out = []
    for K in range(1, max_states + 1):
        objs, LL = fit_best(pops, K, shared_T=False, n_starts=12, n_jobs=n_jobs, rng=rng)
        order_by_lifetime(objs)
        dof = 4 * K + len(pops) * (K * (K - 1) + (K - 1))
        out.append(
            {
                "states": K,
                "LL": float(LL),
                "dof": dof,
                "BIC": float(-2 * LL + np.log(n) * dof),
                "emissions": [e.params.tolist() for e in objs[0].estimate.E],
                "T": [tO.estimate.T.tolist() for tO in objs],
                "pi": [tO.estimate.pi.tolist() for tO in objs],
            }
        )
    return out


def summarize_T(T: np.ndarray) -> dict:
    """Per-dose persistence and memory summaries of transition matrices (doses, K, K)."""
    return {
        "T": T.tolist(),
        "T_tolerant": [float(t[-1, -1]) for t in T],
        "half_life_tolerant": [persistence_half_life(t, t.shape[0] - 1) for t in T],
        "half_life_sensitive": [persistence_half_life(t, 0) for t in T],
        "memory_eigenvalue": [memory_eigenvalue(t) for t in T],
        "memory_half_life": [memory_half_life(t) for t in T],
    }


def percentile_ci(samples: np.ndarray) -> list:
    lo, hi = np.nanpercentile(samples, [2.5, 97.5], axis=0)
    return [np.asarray(lo).tolist(), np.asarray(hi).tolist()]


def run(n_boot: int = 200, n_workers: int = 16, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    pops = [load_lineages(c) for c in CONDITIONS]
    res: dict = {
        "conditions": list(CONDITIONS),
        "doses_nM": list(DOSES_NM),
        "horizon_h": HORIZON_H,
        "cells": [int(sum(len(lin) for lin in pop)) for pop in pops],
        "lineages": [len(pop) for pop in pops],
    }
    obs = [np.vstack([lin.obs for lin in pop]) for pop in pops]
    res["divided"] = [int(np.nansum(o[:, 2] == 1)) for o in obs]
    res["censored"] = [int(np.nansum(o[:, 2] == 0)) for o in obs]

    t0 = time.time()
    res["state_selection"] = select_states(pops, n_jobs=n_workers, rng=rng)

    sweep = dose_sweep(pops, list(DOSES_NM), num_states=2, n_starts=12, n_jobs=n_workers, rng=rng)
    res["LL"] = {k: float(v) for k, v in sweep.LL.items()}
    res["lrt"] = sweep.lrt
    res["emissions"] = [e.params.tolist() for e in sweep.per_dose[0].estimate.E]
    res["mean_lifetime_h"] = [
        e.mean_lifetime() for e in sweep.per_dose[0].estimate.E if isinstance(e, StateDistribution)
    ]
    res["pi"] = [tO.estimate.pi.tolist() for tO in sweep.per_dose]
    res["shared_T"] = sweep.shared[0].estimate.T.tolist()
    res["independent_T"] = [tO.estimate.T.tolist() for tO in sweep.independent]
    res["point"] = summarize_T(sweep.T)

    # Bootstrap over lineages, in parallel chunks.
    chunks = np.array_split(np.arange(n_boot), n_workers)
    seeds = rng.integers(2**32, size=len(chunks))
    with ProcessPoolExecutor(n_workers) as exe:
        futs = [exe.submit(bootstrap_transitions, sweep, pops, len(c), s) for c, s in zip(chunks, seeds, strict=True)]
        boots = np.concatenate([f.result() for f in futs if f.result().size])
    summaries = [summarize_T(b) for b in boots]
    boot: dict = {
        key: percentile_ci(np.array([s[key] for s in summaries], dtype=float))
        for key in ("T_tolerant", "half_life_tolerant", "half_life_sensitive", "memory_eigenvalue", "memory_half_life")
    }
    boot["T"] = percentile_ci(boots)
    # Palbociclib minus control, for tolerant-state persistence and for the memory
    # eigenvalue (which, unlike T_22, does not move just because the drug shifts how
    # many cells are tolerant).
    boot["diff"] = {}
    for key in ("T_tolerant", "memory_eigenvalue"):
        d = np.array([s[key][1] - s[key][0] for s in summaries])
        boot["diff"][key] = {
            "point": res["point"][key][1] - res["point"][key][0],
            "ci": np.percentile(d, [2.5, 97.5]).tolist(),
            "p_two_sided": float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))),
        }
    res["bootstrap"] = boot
    res["n_boot"] = int(boots.shape[0])

    cv = cross_validated_scores(sweep, pops, horizon=HORIZON_H, n_folds=5, escape_col=ESCAPE_COL, rng=rng)
    res["auc"] = auc_table(cv, n_boot=1000, rng=rng)
    pairs = cv["pairs"]
    res["pairs"] = {
        "n": int(pairs["y"].size),
        "escape_fraction_by_dose": [float(np.mean(pairs["y"][pairs["dose"] == d])) for d in range(len(pops))],
        "n_by_dose": [int(np.sum(pairs["dose"] == d)) for d in range(len(pops))],
    }
    # Pooled AUCs are reported for completeness only: escape is far more common in
    # control, so pooling rewards any score that separates the doses.
    res["auc_by_dose"] = auc_by_dose(cv, n_boot=1000, rng=rng)
    res["roc_scores"] = {f: cv[f].tolist() for f in FEATURES}
    res["roc_y"] = pairs["y"].tolist()
    res["roc_dose"] = pairs["dose"].tolist()
    res["runtime_s"] = time.time() - t0
    return res


if __name__ == "__main__":
    n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    out = run(n_boot=n_boot)
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("roc_")}, indent=1, default=float))
