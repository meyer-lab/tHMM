"""Simulation study: recovery of Q, emissions, and ancestral states as leaf sampling drops.

Run with ``uv run python analysis/kptracer/simulation_study.py``. Writes ``results/simulation.csv``.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from lineage import ctmc
from lineage.phylo_sim import simulate_dataset
from lineage.phylo_stats import small_parsimony
from lineage.phyloHMM import fit_best
from lineage.states.StateDistributionGaussian import StateDistribution as Gaussian
from lineage.tree_io import to_lineage

OUT = Path(__file__).parent / "results"

# A progression-like chain: 0 -> 1 -> 2 -> 3 with slow reversions, loosely modeled on the
# AT2-like -> gastric-like -> pre-EMT -> mesenchymal paths reported for KP-Tracer tumors.
RATES = np.array(
    [
        [0.000, 0.150, 0.020, 0.005],
        [0.050, 0.000, 0.100, 0.005],
        [0.005, 0.020, 0.000, 0.080],
        [0.005, 0.005, 0.010, 0.000],
    ]
)
Q_TRUE = ctmc.rates_to_Q(RATES)
PI_TRUE = np.array([0.85, 0.10, 0.04, 0.01])
K, D = 4, 10
N_TREES, N_LEAVES = 20, 500


def match_states(est_means: np.ndarray, true_means: np.ndarray) -> np.ndarray:
    """perm[true state] = estimated state, by minimum total distance between means."""
    C = np.linalg.norm(est_means[:, None, :] - true_means[None, :, :], axis=2)
    r, c = linear_sum_assignment(C)
    perm = np.empty(K, dtype=int)
    perm[c] = r
    return perm


def true_edge_changes(data) -> tuple[np.ndarray, np.ndarray]:
    """Per-tree number of edges of the sampled tree whose endpoint states differ, and edge counts."""
    changes, edges = [], []
    for d in data:
        par = d["tree"].parents[1:]
        s = d["states"]
        changes.append(int(np.sum(s[par] != s[1:])))
        edges.append(len(par))
    return np.array(changes), np.array(edges)


def run_one(args):
    sep, frac, seed = args
    rng = np.random.default_rng(seed)
    means = rng.normal(0, sep, (K, D))
    data = simulate_dataset(N_TREES, N_LEAVES, Q_TRUE, PI_TRUE, means, 1.0, sample_frac=frac, rng=rng)
    X = [to_lineage(d["tree"], d["tree"].leaf_names, d["obs"][d["tree"].is_leaf], [Gaussian(dim=D)]) for d in data]
    n_leaves = int(sum(d["tree"].is_leaf.sum() for d in data))
    tc, n_edges = true_edge_changes(data)
    rows = []

    for mode in ("ctmc", "discrete"):
        if mode == "discrete":
            for x in X:
                x.branch_lengths = None
        m = fit_best(X, K, n_init=3, rng=seed, mode=mode, tol=1e-5)
        perm = match_states(np.array([e.mean for e in m.E]), means)
        leaf_acc, anc_acc = [], []
        for p, d in zip(m.posteriors, data, strict=True):
            est = np.argmax(p.gamma[:, perm], axis=1)
            leaf = d["tree"].is_leaf
            leaf_acc.append(np.mean(est[leaf] == d["states"][leaf]))
            anc_acc.append(np.mean(est[~leaf] == d["states"][~leaf]))
        summ = m.switch_summary()
        est_changes = np.array([s["expected_changes"] for s in summ])
        row = {
            "method": f"tHMM-{mode}",
            "sep": sep,
            "frac": frac,
            "seed": seed,
            "n_leaves": n_leaves,
            "leaf_acc": np.mean(leaf_acc),
            "anc_acc": np.mean(anc_acc),
            "mean_rmse": float(np.sqrt(np.mean((np.array([e.mean for e in m.E])[perm] - means) ** 2))),
            "changes_ratio": est_changes.sum() / tc.sum(),
            "plasticity_corr": np.corrcoef(est_changes / n_edges, tc / n_edges)[0, 1],
            "iters": len(m.LL_trace),
        }
        if mode == "ctmc":
            Qhat = m.Q[np.ix_(perm, perm)]
            off = ~np.eye(K, dtype=bool)
            row["Q_rel_err"] = float(np.linalg.norm(Qhat - Q_TRUE) / np.linalg.norm(Q_TRUE))
            row["exit_rate_rel_err"] = float(np.mean(np.abs(np.diag(Qhat) / np.diag(Q_TRUE) - 1)))
            big = RATES > 0.04
            row["big_rate_rel_err"] = float(np.mean(np.abs(Qhat[big] / RATES[big] - 1)))
            row["small_rate_abs_err"] = float(np.mean(np.abs(Qhat[off & ~big] - RATES[off & ~big])))
            row["pi_tv"] = 0.5 * float(np.abs(m.pi[perm] - PI_TRUE).sum())
            for i, j in zip(*np.nonzero(off), strict=True):
                row[f"q{i}{j}"] = Qhat[i, j]
        rows.append(row)

    # Cluster-then-parsimony baseline, with k-means labels and with the true leaf labels (oracle)
    obs = np.vstack([d["obs"][d["tree"].is_leaf] for d in data])
    km = KMeans(K, n_init=5, random_state=seed).fit(obs)
    perm = match_states(km.cluster_centers_, means)
    inv = np.argsort(perm)  # estimated label -> true label
    for label_source in ("kmeans", "oracle"):
        leaf_acc, anc_acc, changes = [], [], []
        if label_source == "kmeans":
            lab_iter = iter(inv[km.labels_])
        for d in data:
            leaf = d["tree"].is_leaf
            lab = np.full(len(leaf), -1)
            if label_source == "kmeans":
                lab[leaf] = [next(lab_iter) for _ in range(leaf.sum())]
            else:
                lab[leaf] = d["states"][leaf]
            score, labels, _ = small_parsimony(d["tree"], lab, K)
            leaf_acc.append(np.mean(labels[leaf] == d["states"][leaf]))
            anc_acc.append(np.mean(labels[~leaf] == d["states"][~leaf]))
            changes.append(score)
        changes = np.array(changes)
        rows.append(
            {
                "method": f"parsimony-{label_source}",
                "sep": sep,
                "frac": frac,
                "seed": seed,
                "n_leaves": n_leaves,
                "leaf_acc": np.mean(leaf_acc),
                "anc_acc": np.mean(anc_acc),
                "changes_ratio": changes.sum() / tc.sum(),
                "plasticity_corr": np.corrcoef(changes / n_edges, tc / n_edges)[0, 1],
            }
        )
    return rows


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    grid = list(itertools.product([1.0, 0.5], [1.0, 0.5, 0.25, 0.1], range(5)))
    with ProcessPoolExecutor(40) as ex:
        rows = [r for res in ex.map(run_one, grid) for r in res]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "simulation.csv", index=False)
    cols = ["leaf_acc", "anc_acc", "changes_ratio", "plasticity_corr", "Q_rel_err", "big_rate_rel_err", "pi_tv"]
    print(df.groupby(["sep", "frac", "method"])[cols].mean().round(3).to_string())
