"""Heritability of the CDK2 state under timed MEK or ERK inhibition (S-BSST314 Figure 1).

Run as ``OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m lineage.mitogen_analysis
[n_boot]``. For each drug, the six exposures (none, 1, 3, 6, 9 h, left on) are fit as a
dose series by :func:`.heritability.dose_sweep`, with lineage roots' lifetimes truncated
to the window in which they were selected (see :func:`.palbociclib_loader.build_lineages`).
Results are written to ``output/mitogen_analysis.json`` and drawn by
:mod:`lineage.figures.figure23`.

See :mod:`lineage.mitogen_loader` for the data.
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .heritability import dose_sweep
from .mitogen_loader import DRUGS, PULSES_H, generation_depths, load_lineages
from .palbociclib_analysis import bootstrap_summary, division_calibration, log, summarize_T
from .palbociclib_loader import ROOT_MODES
from .states.CensoredWeibullGaussian import StateDistribution

OUTPUT = os.path.join("output", "mitogen_analysis.json")

#: Lineages kept per condition. The movies hold 2,000-18,000 lineages per condition, and
#: EM time is dominated by per-lineage overhead, so each condition is subsampled to about
#: the size of the palbociclib data (the stay-in MEKi wells have fewer and are kept whole).
MAX_LINEAGES = 2500


def condition_lineages(drug: str, rng, roots: tuple[str, ...] = ROOT_MODES) -> tuple[dict[str, list[list]], list[int]]:
    """Per-condition lineages under each root treatment, with the same random subset of
    lineages for every treatment, and how many lineages each condition had to draw from."""
    out: dict[str, list[list]] = {mode: [] for mode in roots}
    available = []
    for p in PULSES_H:
        full = {mode: load_lineages(drug, p, roots=mode) for mode in roots}
        n = len(full[roots[0]])
        available.append(n)
        keep = np.sort(rng.choice(n, size=min(n, MAX_LINEAGES), replace=False))
        for mode in roots:
            out[mode].append([full[mode][i] for i in keep])
    return out, available


def relative_correlations(lineages: list) -> dict[str, dict[str, float]]:
    """Model-free check on inheritance: correlation of the CDK2 activation rate between
    mother-daughter, sister, and first-cousin pairs born into the condition.

    Under a hidden state with memory, sisters (one division apart from a shared mother)
    correlate more than cousins (two apart from a shared grandmother).
    """
    pairs: dict[str, list] = {"mother_daughter": [], "sisters": [], "cousins": []}
    for lin in lineages:
        x = lin.obs[:, 0]
        parents, daughters = lin.edges
        mother = np.full(len(lin), -1)
        mother[daughters] = parents
        kids: dict[int, list[int]] = {}
        for p, d in zip(parents, daughters, strict=True):
            kids.setdefault(int(p), []).append(int(d))
            if p > 0:  # the root's G1 was before the drug
                pairs["mother_daughter"].append((x[p], x[d]))
        for sibs in kids.values():
            if len(sibs) == 2:
                pairs["sisters"].append((x[sibs[0]], x[sibs[1]]))
                a, b = (kids.get(s, []) for s in sibs)
                pairs["cousins"] += [(x[i], x[j]) for i in a for j in b]
    out = {}
    for key, v in pairs.items():
        arr = np.array(v, dtype=float).reshape(-1, 2)
        arr = arr[np.all(np.isfinite(arr), axis=1)]
        r = float(np.corrcoef(arr.T)[0, 1]) if arr.shape[0] > 2 else np.nan
        out[key] = {"r": r, "n": int(arr.shape[0])}
    return out


def fit_drug(drug: str, n_boot: int, n_workers: int, rng, t0: float) -> dict:
    lineages, available = condition_lineages(drug, rng)
    pops, full = lineages["truncate"], lineages["keep"]
    res: dict = {
        "lineages_available": available,
        "cells": [int(sum(len(lin) for lin in pop)) for pop in pops],
        "lineages": [len(pop) for pop in pops],
        "depth_counts": [np.bincount(generation_depths(pop)).tolist() for pop in pops],
        "relatives": [relative_correlations(pop) for pop in pops],
    }
    sweep = dose_sweep(pops, list(PULSES_H), num_states=2, n_starts=8, n_jobs=n_workers, rng=rng)
    log(f"{drug}: dose sweep done", t0)
    res["LL"] = {k: float(v) for k, v in sweep.LL.items()}
    res["lrt"] = sweep.lrt
    res["emissions"] = [e.params.tolist() for e in sweep.per_dose[0].estimate.E]
    res["mean_lifetime_h"] = [
        e.mean_lifetime() for e in sweep.per_dose[0].estimate.E if isinstance(e, StateDistribution)
    ]
    res["pi"] = [tO.estimate.pi.tolist() for tO in sweep.per_dose]
    res["independent_T"] = [tO.estimate.T.tolist() for tO in sweep.independent]
    res["point"] = summarize_T(sweep.T)

    res["divided_by_20h"] = division_calibration(full, sweep)
    res["bootstrap"], res["n_boot"] = bootstrap_summary(sweep, pops, n_boot, n_workers, rng, res["point"])
    log(f"{drug}: bootstrap done", t0)

    # The same sweep with the roots' lifetimes taken at face value or dropped.
    res["root_modes"] = {}
    for mode in ROOT_MODES[1:]:
        alt = dose_sweep(
            lineages[mode],
            list(PULSES_H),
            num_states=2,
            n_starts=4,
            n_jobs=n_workers,
            rng=rng,
        )
        res["root_modes"][mode] = {
            "lrt": alt.lrt,
            "emissions": [e.params.tolist() for e in alt.per_dose[0].estimate.E],
            "point": summarize_T(alt.T),
            "divided_by_20h": division_calibration(full, alt),
        }
        log(f"{drug}: roots={mode} done", t0)
    return res


def run(n_boot: int = 96, n_workers: int = 44, seed: int = 0) -> dict:
    """Fit the drugs concurrently, each with its own share of the worker processes (the
    fits themselves run in those processes, so threads are enough to overlap them)."""
    t0 = time.time()
    rngs = np.random.default_rng(seed).spawn(len(DRUGS))
    with ThreadPoolExecutor(len(DRUGS)) as exe:
        futs = {
            d: exe.submit(fit_drug, d, n_boot, n_workers // len(DRUGS), r, t0) for d, r in zip(DRUGS, rngs, strict=True)
        }
    out: dict = {"drugs": list(DRUGS), "pulses_h": [float(p) for p in PULSES_H]}
    out |= {d: f.result() for d, f in futs.items()}
    out["runtime_s"] = time.time() - t0
    return out


if __name__ == "__main__":
    out = run(n_boot=int(sys.argv[1]) if len(sys.argv) > 1 else 96)
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps({d: {k: out[d][k] for k in ("lrt", "point")} for d in DRUGS}, indent=1, default=float))
