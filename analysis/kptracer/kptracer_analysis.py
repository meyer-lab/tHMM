"""Fit the branch-length tHMM to KP-Tracer tumors and compare it with parsimony, PATH and published scores.

Stages (run in order; each writes into ``results/``)::

    uv run python analysis/kptracer/kptracer_analysis.py select    # K and branch-length model selection
    uv run python analysis/kptracer/kptracer_analysis.py fit       # final unsupervised + cluster-anchored fits
    uv run python analysis/kptracer/kptracer_analysis.py compare   # per-tumor benchmarks

Requires ``results/kptracer_inputs.pkl`` from ``prepare_data.py``.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from lineage.phylo_stats import (
    leaf_node_distance,
    phylogenetic_correlation,
    phylogenetic_correlation_z,
    small_parsimony,
)
from lineage.phyloHMM import PhyloHMM, fit_best, mask_leaves
from lineage.states.StateDistributionGaussian import StateDistribution as Gaussian
from lineage.tree_io import to_lineage

RES = Path(__file__).parent / "results"
MAIN_DELTA = 0.5


def load_records(exclude=()):
    with open(RES / "kptracer_inputs.pkl", "rb") as f:
        recs = pickle.load(f)
    return [r for r in recs if r["tumor"] not in exclude]


def branch_lengths(rec, model: str, delta: float) -> np.ndarray:
    """``mut``: mutations on the edge + delta. ``unit``: every edge has length 1."""
    if model == "unit":
        bl = np.ones(len(rec["n_mut"]))
    else:
        bl = rec["n_mut"] + delta
    bl[0] = 0.0
    return bl


def build(recs, model="mut", delta=MAIN_DELTA, emb="scvi"):
    D = recs[0][emb].shape[1]
    return [
        to_lineage(
            r["tree"], r["leaf_names"], r[emb], [Gaussian(dim=D)], branch_lengths=branch_lengths(r, model, delta)
        )
        for r in recs
    ]


# ---------------------------------------------------------------------------- stage 1: selection


def select_job(args):
    K, rep, model, delta, exclude = args
    recs = load_records(exclude)
    X = build(recs, model, delta)
    Xm = mask_leaves(X, 0.2, rng=1000 + rep)
    m = fit_best(Xm, K, n_init=2, rng=rep, tol=1e-4, max_iter=200)
    m_full = fit_best(X, K, n_init=2, rng=rep, tol=1e-4, max_iter=200) if rep == 0 else None
    row = {
        "K": K,
        "rep": rep,
        "bl_model": model,
        "delta": delta,
        "excluded": ",".join(exclude),
        "heldout_ll": m.heldout_loglik(Xm, X),
        "heldout_ll_no_tree": m.heldout_loglik(Xm, X, lineage=False),
        "train_ll": m.LL,
    }
    if m_full is not None:
        row.update({"full_ll": m_full.LL, "BIC": m_full.BIC(), "n_params": m_full.num_parameters()})
    print(row, flush=True)
    return row


def stage_select():
    jobs = [(K, rep, "mut", MAIN_DELTA, ()) for K in range(2, 15) for rep in range(3)]
    # Branch-length sensitivity at a few K
    for K in (6, 8, 10):
        for model, delta in [("mut", 0.1), ("mut", 0.25), ("mut", 1.0), ("mut", 2.0), ("unit", 0.0)]:
            jobs += [(K, rep, model, delta, ()) for rep in range(3)]
    with ProcessPoolExecutor(46) as ex:
        rows = list(ex.map(select_job, jobs))
    pd.DataFrame(rows).to_csv(RES / "model_selection.csv", index=False)


# ---------------------------------------------------------------------------- stage 2: fits


def fit_job(args):
    name, K, exclude, anchored, delta, model, emb = args
    recs = load_records(exclude)
    X = build(recs, model, delta, emb)
    if anchored:
        # One state per published Leiden cluster present after filtering, with emissions started at
        # the cluster means; EM is then free to move them.
        labels = np.concatenate([r["leiden"] for r in recs])
        uniq = np.unique(labels)
        obs = np.vstack([r[emb] for r in recs])
        E = []
        for c in uniq:
            g = Gaussian(dim=obs.shape[1])
            g.estimator(obs, (labels == c).astype(float))
            E.append(g)
        # By default the emissions stay fixed at the cluster-conditional Gaussians, so every state keeps
        # its published identity; with free emissions EM relabels states (see the report).
        m = PhyloHMM(X, len(uniq), rng=0, fix_E=None if anchored == "free" else E)
        if anchored == "free":
            m.E = E
        m.fit(init=False, tol=1e-5, max_iter=400)
        state_labels = uniq
    else:
        m = fit_best(X, K, n_init=5, rng=0, tol=1e-5, max_iter=400)
        state_labels = None
    out = {
        "name": name,
        "tumors": [r["tumor"] for r in recs],
        "Q": m.Q,
        "pi": m.pi,
        "means": np.array([e.mean for e in m.E]),
        "vars": np.array([e.var for e in m.E]),
        "LL": m.LL,
        "BIC": m.BIC(),
        "gammas": [p.gamma for p in m.posteriors],
        "summary": m.switch_summary(),
        "state_labels": state_labels,
        "iters": len(m.LL_trace),
        "delta": delta,
        "bl_model": model,
        "emb": emb,
        "fixed_E": bool(anchored) and anchored != "free",
    }
    with open(RES / f"fit_{name}.pkl", "wb") as f:
        pickle.dump(out, f)
    print(name, m.LL, len(m.LL_trace), flush=True)
    return name


def stage_fit(K: int, anchored_only: bool = False):
    jobs = [
        ("anchored", None, (), True, MAIN_DELTA, "mut", "scvi"),
        ("anchored_no3724", None, ("3724_NT_T1",), True, MAIN_DELTA, "mut", "scvi"),
        ("anchored_delta0.1", None, (), True, 0.1, "mut", "scvi"),
        ("anchored_delta2", None, (), True, 2.0, "mut", "scvi"),
        ("anchored_unit", None, (), True, 0.0, "unit", "scvi"),
        ("anchored_freeE", None, (), "free", MAIN_DELTA, "mut", "scvi"),
    ]
    if not anchored_only:
        jobs.append(("unsup", K, (), False, MAIN_DELTA, "mut", "scvi"))
        jobs.append(("unsup_no3724", K, ("3724_NT_T1",), False, MAIN_DELTA, "mut", "scvi"))
        jobs.append(("unsup_pca", K, (), False, MAIN_DELTA, "mut", "pca"))
    with ProcessPoolExecutor(len(jobs)) as ex:
        list(ex.map(fit_job, jobs))


# ---------------------------------------------------------------------------- stage 2b: bootstrap


def bootstrap_job(args):
    b, name = args
    with open(RES / f"fit_{name}.pkl", "rb") as f:
        fit = pickle.load(f)
    recs = [r for r in load_records() if r["tumor"] in fit["tumors"]]
    rng = np.random.default_rng(b)
    pick = rng.integers(len(recs), size=len(recs))
    X = build([recs[i] for i in pick], fit["bl_model"], fit["delta"], fit["emb"])
    E = [Gaussian(mean=mu, var=v) for mu, v in zip(fit["means"], fit["vars"], strict=True)]
    m = PhyloHMM(X, len(fit["pi"]), rng=b, fix_E=E if fit.get("fixed_E") else None)
    m.E = [Gaussian(mean=mu, var=v) for mu, v in zip(fit["means"], fit["vars"], strict=True)]
    m.Q, m.pi = fit["Q"].copy(), fit["pi"].copy()
    m.fit(init=False, tol=1e-5, max_iter=400)
    summ = m.switch_summary()
    joint = sum(s["edge_joint"] for s in summ)
    jumps = sum(s["jumps"] for s in summ)
    return {
        "b": b,
        "Q": m.Q,
        "pi": m.pi,
        "stay": np.diag(joint) / joint.sum(axis=1),
        "jumps": jumps,
        "root": np.mean([p.gamma[0] for p in m.posteriors], axis=0),
    }


def stage_bootstrap(name="anchored", B=48):
    with ProcessPoolExecutor(min(B, 46)) as ex:
        res = list(ex.map(bootstrap_job, [(b, name) for b in range(B)]))
    with open(RES / f"bootstrap_{name}.pkl", "wb") as f:
        pickle.dump(res, f)


# ---------------------------------------------------------------------------- stage 3: compare


def tumor_benchmarks(fit, recs, max_path_leaves=4000, rng=0):
    """Per-tumor tHMM plasticity vs Fitch parsimony, PATH auto-correlation, and published scPlasticity."""
    rng = np.random.default_rng(rng)
    rows, cell_rows = [], []
    for r, gam, summ in zip(recs, fit["gammas"], fit["summary"], strict=True):
        pt = r["tree"]
        leaf = pt.is_leaf
        uniq = np.unique(r["leiden"])
        lab_small = np.searchsorted(uniq, r["leiden"])
        lab = np.full(len(pt.names), -1)
        lab[leaf] = lab_small
        score, _, _ = small_parsimony(pt, lab, len(uniq))
        n_edges = len(pt.names) - 1

        # PATH-style Moran's I of the Leiden labels, averaged over states weighted by frequency
        leaves = np.nonzero(leaf)[0]
        sel = np.arange(leaves.size)
        if leaves.size > max_path_leaves:
            sel = np.sort(rng.choice(leaves.size, max_path_leaves, replace=False))
        dist = leaf_node_distance(pt, leaves[sel])
        C, Cz = phylogenetic_correlation_z(lab_small[sel], dist, len(uniq), n_perm=100, rng=rng)
        freq = np.bincount(lab_small[sel], minlength=len(uniq)) / sel.size
        path_auto = float(np.sum(np.diag(C) * freq))
        path_auto_z = float(np.sum(np.diag(Cz) * freq))

        # Same Moran's I on tHMM MAP states, to separate the state definition from the method
        map_state = np.argmax(gam[leaf], axis=1)
        C2 = phylogenetic_correlation(map_state[sel], dist, gam.shape[1])
        f2 = np.bincount(map_state[sel], minlength=gam.shape[1]) / sel.size
        path_auto_thmm = float(np.nansum(np.diag(C2) * f2))

        # Per-cell tHMM plasticity: mean edge-change probability along the root-to-leaf path
        par = pt.parents
        pe = np.zeros(len(pt.names))
        pe[summ_children(pt)] = summ["per_edge_change"]
        cum = np.zeros(len(pt.names))
        nedge = np.zeros(len(pt.names))
        for i in range(1, len(pt.names)):
            cum[i] = cum[par[i]] + pe[i]
            nedge[i] = nedge[par[i]] + 1
        cell_score = cum[leaf] / nedge[leaf]
        pub = r["published_scPlasticity"]
        ok = np.isfinite(pub)
        cell_rows.append(
            pd.DataFrame(
                {"tumor": r["tumor"], "cell": r["leaf_names"], "thmm_cell_plasticity": cell_score, "published": pub}
            )
        )
        rows.append(
            {
                "tumor": r["tumor"],
                "n_cells": int(leaf.sum()),
                "n_edges": n_edges,
                "n_clusters": len(uniq),
                "fitch_changes": score,
                "fitch_plasticity": score / n_edges,
                "thmm_changes": summ["expected_changes"],
                "thmm_plasticity": summ["plasticity"],
                "thmm_jumps": float(summ["jumps"].sum()),
                "path_autocorr_leiden": path_auto,
                "path_autocorr_z_leiden": path_auto_z,
                "path_autocorr_thmm": path_auto_thmm,
                "published_mean_scPlasticity": float(np.nanmean(pub)) if ok.any() else np.nan,
                "cell_spearman_vs_published": stats.spearmanr(cell_score[ok], pub[ok])[0] if ok.sum() > 10 else np.nan,
            }
        )
    return pd.DataFrame(rows), pd.concat(cell_rows)


def summ_children(pt):
    """Children in CSR edge order, which is the order of ``per_edge_change``."""
    return pt.tree.indices


def pooled_transition_tables(fit, recs):
    """Pooled k -> l change counts: tHMM expected edge changes and CTMC jumps (anchored states), and Fitch."""
    labels = fit["state_labels"]
    K = len(labels)
    thmm = sum(s["edge_changes"] for s in fit["summary"])
    jumps = sum(s["jumps"] for s in fit["summary"])
    fitch = np.zeros((K, K))
    for r in recs:
        pt = r["tree"]
        lab = np.full(len(pt.names), -1)
        lab[pt.is_leaf] = np.searchsorted(labels, r["leiden"])
        _, _, counts = small_parsimony(pt, lab, K)
        fitch += counts
    return thmm, jumps, fitch


def state_table(fit, recs, jumps):
    """Per-state stability from the anchored fit, next to Fitch and the published per-cell scores."""
    labels = fit["state_labels"]
    K = len(labels)
    names = {}
    for r in recs:
        names.update(dict(zip(r["leiden"].tolist(), r["cluster_name"].tolist(), strict=True)))
    joint = sum(s["edge_joint"] for s in fit["summary"])  # pooled P(parent = k, child = l)
    fitch_edges = np.zeros((K, K))
    root = np.zeros(K)
    n_cells = np.zeros(K)
    pub: dict[int, list[float]] = {k: [] for k in range(K)}
    for r, gam in zip(recs, fit["gammas"], strict=True):
        pt = r["tree"]
        leaf_lab = np.searchsorted(labels, r["leiden"])
        lab = np.full(len(pt.names), -1)
        lab[pt.is_leaf] = leaf_lab
        _, fl, _ = small_parsimony(pt, lab, K)
        np.add.at(fitch_edges, (fl[pt.parents[1:]], fl[1:]), 1)
        root += gam[0]
        n_cells += np.bincount(leaf_lab, minlength=K)
        for k in range(K):
            v = r["published_scPlasticity"][leaf_lab == k]
            v = v[np.isfinite(v)]
            if v.size:
                pub[k].append(float(v.mean()))
    Q = fit["Q"]
    return pd.DataFrame(
        {
            "leiden": labels,
            "name": [names.get(int(k), str(k)) for k in labels],
            "n_cells": n_cells.astype(int),
            "exit_rate": -np.diag(Q),
            "thmm_stay_prob": np.diag(joint) / np.maximum(joint.sum(axis=1), 1e-12),
            "fitch_stay_prob": np.diag(fitch_edges) / np.maximum(fitch_edges.sum(axis=1), 1),
            # Median over tumors of the per-tumor mean, as in Yang et al. (higher = more plastic)
            "published_scPlasticity": [np.median(pub[k]) if pub[k] else np.nan for k in range(K)],
            "root_posterior": root / len(recs),
            "stationary": _stationary(Q),
            "net_outflow_jumps": jumps.sum(axis=1) - jumps.sum(axis=0),
        }
    )


def _stationary(Q):
    from lineage.ctmc import stationary

    return stationary(Q)


def bootstrap_summary(name="anchored"):
    """95% tumor-bootstrap intervals for per-state stability and for the direction of the largest flows."""
    path = RES / f"bootstrap_{name}.pkl"
    if not path.exists():
        return
    with open(path, "rb") as f:
        boot = pickle.load(f)
    st = pd.read_csv(RES / f"state_table_{name}.csv")
    stay = np.array([b["stay"] for b in boot])
    exit_rate = np.array([-np.diag(b["Q"]) for b in boot])
    root = np.array([b["root"] for b in boot])
    for key, arr in (("stay", stay), ("exit_rate", exit_rate), ("root", root)):
        st[f"{key}_lo"] = np.nanpercentile(arr, 2.5, axis=0)
        st[f"{key}_hi"] = np.nanpercentile(arr, 97.5, axis=0)
    st["stay_rank_median"] = np.median(np.argsort(np.argsort(-stay, axis=1), axis=1), axis=0) + 1
    st.to_csv(RES / f"state_table_{name}.csv", index=False)

    J = pd.read_csv(RES / f"transitions_jumps_{name}.csv", index_col=0).values
    jumps = np.array([b["jumps"] for b in boot])
    names = st["name"].tolist()
    rows = []
    for i, j in zip(*np.triu_indices(len(names), 1), strict=True):
        total = J[i, j] + J[j, i]
        if total < 50:
            continue
        net = jumps[:, i, j] - jumps[:, j, i]
        frac = jumps[:, i, j] / np.maximum(jumps[:, i, j] + jumps[:, j, i], 1e-12)
        a, b = (i, j) if J[i, j] >= J[j, i] else (j, i)
        rows.append(
            {
                "from": names[a],
                "to": names[b],
                "jumps_fwd": J[a, b],
                "jumps_rev": J[b, a],
                "fwd_fraction": J[a, b] / (J[a, b] + J[b, a]),
                "fwd_fraction_lo": np.percentile(frac if a == i else 1 - frac, 2.5),
                "fwd_fraction_hi": np.percentile(frac if a == i else 1 - frac, 97.5),
                "boot_same_direction": float(np.mean(np.sign(net) == np.sign(J[i, j] - J[j, i]))),
            }
        )
    pd.DataFrame(rows).sort_values("jumps_fwd", ascending=False).to_csv(RES / f"flows_{name}.csv", index=False)


def stage_compare():
    out = {}
    for name in (
        "anchored",
        "unsup",
        "anchored_no3724",
        "anchored_delta0.1",
        "anchored_delta2",
        "anchored_unit",
        "anchored_freeE",
    ):
        path = RES / f"fit_{name}.pkl"
        if not path.exists():
            continue
        with open(path, "rb") as f:
            fit = pickle.load(f)
        recs = [r for r in load_records() if r["tumor"] in fit["tumors"]]
        tb, cells = tumor_benchmarks(fit, recs)
        tb.to_csv(RES / f"tumor_benchmarks_{name}.csv", index=False)
        if name in ("anchored", "unsup"):
            cells.to_csv(RES / f"cell_plasticity_{name}.csv.gz", index=False)
        pairs = [
            ("thmm_plasticity", "fitch_plasticity"),
            ("thmm_plasticity", "published_mean_scPlasticity"),
            ("fitch_plasticity", "published_mean_scPlasticity"),
            ("thmm_plasticity", "path_autocorr_leiden"),
            ("fitch_plasticity", "path_autocorr_leiden"),
            ("thmm_plasticity", "path_autocorr_thmm"),
            ("thmm_plasticity", "path_autocorr_z_leiden"),
            ("fitch_plasticity", "path_autocorr_z_leiden"),
        ]
        out[name] = {f"{a}~{b}": stats.spearmanr(tb[a], tb[b])[0] for a, b in pairs}
        allc = cells.dropna()
        out[name]["cell_pooled_spearman"] = stats.spearmanr(allc.thmm_cell_plasticity, allc.published)[0]
        out[name]["cell_median_within_tumor_spearman"] = float(np.nanmedian(tb.cell_spearman_vs_published))
        if fit["state_labels"] is not None:
            thmm, jumps, fitch = pooled_transition_tables(fit, recs)
            lab = [str(x) for x in fit["state_labels"]]
            pd.DataFrame(thmm, index=lab, columns=lab).to_csv(RES / f"transitions_thmm_{name}.csv")
            pd.DataFrame(jumps, index=lab, columns=lab).to_csv(RES / f"transitions_jumps_{name}.csv")
            pd.DataFrame(fitch, index=lab, columns=lab).to_csv(RES / f"transitions_fitch_{name}.csv")
            off = ~np.eye(len(lab), dtype=bool)
            out[name]["transition_matrix_spearman_thmm_vs_fitch"] = stats.spearmanr(thmm[off], fitch[off])[0]
            out[name]["total_changes_thmm"] = float(thmm.sum())
            out[name]["total_changes_fitch"] = float(fitch.sum())
            out[name]["total_jumps_ctmc"] = float(jumps.sum())
            st = state_table(fit, recs, jumps)
            st.to_csv(RES / f"state_table_{name}.csv", index=False)
            for a, b in [("thmm_stay_prob", "fitch_stay_prob"), ("thmm_stay_prob", "published_scPlasticity")]:
                out[name][f"state_{a}~{b}"] = stats.spearmanr(st[a], st[b])[0]
    pd.DataFrame(out).T.to_csv(RES / "benchmark_correlations.csv")
    bootstrap_summary("anchored")
    print(pd.DataFrame(out).T.round(3).to_string())


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "select":
        stage_select()
    elif stage == "fit":
        stage_fit(int(sys.argv[2]))
    elif stage == "fit_anchored":
        stage_fit(0, anchored_only=True)
    elif stage == "bootstrap":
        stage_bootstrap(*sys.argv[2:3])
    elif stage == "compare":
        stage_compare()
