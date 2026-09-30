"""tHMM on the 100-leaf tree used for the BDMM-Prime comparison, with the same tip types as data.

Emissions are fixed near-indicators of the observed type, so the tHMM sees exactly what BDMM-Prime
sees (tree, branch lengths, tip types). Intervals come from a parametric bootstrap. Writes
``results/bdmm_thmm.json``. Inputs are in ``bdmm/`` (regenerate them with ``--make-tree``), together
with the BDMM-Prime XML and its posterior summary (BEAST 2.7.7, BDMM-Prime 2.7.2).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lineage.phylo_sim import simulate_states, yule_tree
from lineage.phyloHMM import PhyloHMM
from lineage.states.StateDistributionGaussian import StateDistribution as Gaussian
from lineage.tree_io import graph_to_phylotree, load_tree, to_lineage

BDMM = Path(__file__).parent / "bdmm"
RES = Path(__file__).parent / "results"
K = 4


def fit(pt, types, rng):
    onehot = np.eye(K)[types]
    E = [Gaussian(mean=np.eye(K)[k], var=np.full(K, 0.01)) for k in range(K)]
    lin = to_lineage(pt, pt.leaf_names, onehot, E, branch_lengths=pt.branch_lengths)
    return PhyloHMM([lin], K, rng=rng, fix_E=E).fit(init=False, tol=1e-7, max_iter=2000)


def make_tree():
    """The 100-leaf Yule tree and tip types given to both methods (rates 3x the simulation study's)."""
    from simulation_study import PI_TRUE, Q_TRUE

    rng = np.random.default_rng(11)
    Q = Q_TRUE * 3
    pt = graph_to_phylotree(yule_tree(100, rng=rng))
    s = simulate_states(pt, Q, PI_TRUE, rng)

    def nwk(i):
        ch = pt.tree.indices[pt.tree.indptr[i] : pt.tree.indptr[i + 1]]
        inner = "(" + ",".join(nwk(c) for c in ch) + ")" if len(ch) else ""
        return f"{inner}{pt.names[i]}:{pt.branch_lengths[i]:.6f}"

    BDMM.mkdir(exist_ok=True)
    (BDMM / "tree100.nwk").write_text(nwk(0) + ";\n")
    tips = [f"{n}\t{st}" for n, st, leaf in zip(pt.names, s, pt.is_leaf, strict=True) if leaf]
    (BDMM / "tip_types.tsv").write_text("taxon\ttype\n" + "\n".join(tips) + "\n")
    truth = {"Q": Q.tolist(), "pi": PI_TRUE.tolist(), "birth_rate": 1.0}
    truth["all_states"] = dict(zip(map(str, pt.names), map(int, s), strict=True))
    (BDMM / "truth.json").write_text(json.dumps(truth))


if __name__ == "__main__":
    if "--make-tree" in sys.argv:
        make_tree()
        sys.exit()
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
