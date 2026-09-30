"""tHMM on the 100-leaf tree used for the BDMM-Prime comparison, with the same tip types as data.

Emissions are fixed near-indicators of the observed type, so the tHMM sees exactly what BDMM-Prime
sees (tree, branch lengths, tip types). Intervals come from a parametric bootstrap. Writes
``results/bdmm_thmm.json``. Inputs are in ``$BDMM_DIR`` (default ``~/data/bdmm``).
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lineage.phylo_sim import simulate_states
from lineage.phyloHMM import PhyloHMM
from lineage.states.StateDistributionGaussian import StateDistribution as Gaussian
from lineage.tree_io import load_tree, to_lineage

BDMM = Path(os.environ.get("BDMM_DIR", Path.home() / "data/bdmm"))
RES = Path(__file__).parent / "results"
K = 4


def fit(pt, types, rng):
    onehot = np.eye(K)[types]
    E = [Gaussian(mean=np.eye(K)[k], var=np.full(K, 0.01)) for k in range(K)]
    lin = to_lineage(pt, pt.leaf_names, onehot, E, branch_lengths=pt.branch_lengths)
    return PhyloHMM([lin], K, rng=rng, fix_E=E).fit(init=False, tol=1e-7, max_iter=2000)


if __name__ == "__main__":
    truth = json.loads((BDMM / "truth.json").read_text())
    pt = load_tree(BDMM / "tree100.nwk", collapse_unifurcations=False)
    tips = pd.read_csv(BDMM / "tip_types.tsv", sep="\t", index_col=0)["type"]
    types = tips.loc[list(pt.leaf_names)].values.astype(int)

    t0 = time.perf_counter()
    m = fit(pt, types, 0)
    wall = time.perf_counter() - t0

    rng = np.random.default_rng(1)
    boots = []
    t0 = time.perf_counter()
    for _ in range(200):
        s = simulate_states(pt, m.Q, m.pi, rng)
        boots.append(fit(pt, s[pt.is_leaf], rng).Q)
    boot_wall = time.perf_counter() - t0
    boots = np.array(boots)
    Qtrue = np.array(truth["Q"])
    off = ~np.eye(K, dtype=bool)
    lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
    out = {
        "fit_seconds": wall,
        "bootstrap_seconds_200": boot_wall,
        "rates_true": Qtrue.tolist(),
        "rates_mle": m.Q.tolist(),
        "rates_lo": lo.tolist(),
        "rates_hi": hi.tolist(),
        "coverage": float(np.mean((Qtrue[off] >= lo[off]) & (Qtrue[off] <= hi[off]))),
        "median_abs_log_ratio": float(np.median(np.abs(np.log(m.Q[off] / np.maximum(Qtrue[off], 1e-6))))),
    }
    (RES / "bdmm_thmm.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("rates")}, indent=1))
    print(np.round(m.Q, 3))
    print(np.round(Qtrue, 3))
