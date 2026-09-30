"""Figures for the KP-Tracer report. Reads ``results/`` and writes PNGs to ``figures/``."""

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RES = HERE / "results"
FIG = HERE / "figures"

# Categorical slots in fixed order (reference palette, light mode)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
METHODS = {
    "tHMM-ctmc": ("tHMM, exp(Qt) per edge", C[0]),
    "tHMM-discrete": ("tHMM, one T per edge", C[1]),
    "parsimony-kmeans": ("k-means + parsimony", C[2]),
    "parsimony-oracle": ("true labels + parsimony", C[3]),
}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "figure.dpi": 150,
    }
)


def fig_simulation():
    df = pd.read_csv(RES / "simulation.csv")
    metrics = [
        ("leaf_acc", "Leaf state accuracy"),
        ("anc_acc", "Ancestral state accuracy"),
        ("changes_ratio", "Estimated / true state changes"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharex=True)
    for row, sep in enumerate([1.0, 0.5]):
        sub = df[df.sep == sep]
        for col, (m, title) in enumerate(metrics):
            ax = axes[row, col]
            for key, (label, color) in METHODS.items():
                g = sub[sub.method == key].groupby("frac")[m]
                mu, sd = g.mean(), g.std()
                ax.errorbar(
                    mu.index, mu.values, yerr=sd.values, color=color, lw=2, marker="o", ms=4, capsize=2, label=label
                )
            ax.set_xscale("log")
            ax.set_xticks([0.1, 0.25, 0.5, 1.0], ["10%", "25%", "50%", "100%"])
            if m == "changes_ratio":
                ax.axhline(1.0, color=INK2, lw=0.8, ls="--")
            ax.set_title(f"{title}\n{'well separated' if sep == 1.0 else 'overlapping'} states", fontsize=9)
            if row == 1:
                ax.set_xlabel("Leaves sampled")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "simulation_accuracy.png")

    ct = df[df.method == "tHMM-ctmc"]
    fig, ax = plt.subplots(figsize=(4.2, 3))
    for i, sep in enumerate([1.0, 0.5]):
        g = ct[ct.sep == sep].groupby("frac")
        ax.errorbar(
            g.Q_rel_err.mean().index,
            g.Q_rel_err.mean(),
            yerr=g.Q_rel_err.std(),
            color=C[i],
            lw=2,
            marker="o",
            ms=4,
            capsize=2,
            label=f"Q, {'well separated' if sep == 1 else 'overlapping'}",
        )
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.25, 0.5, 1.0], ["10%", "25%", "50%", "100%"])
    ax.set_xlabel("Leaves sampled")
    ax.set_ylabel("‖Q̂ − Q‖ / ‖Q‖")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "simulation_Q_error.png")


def fig_scalability():
    df = pd.read_csv(RES / "scalability.csv")
    fig, ax = plt.subplots(figsize=(4.2, 3))
    ax.loglog(df.n_nodes, df.estep_s, color=C[0], lw=2, marker="o", ms=4, label="one E-step")
    ax.loglog(df.n_nodes, df.fit_s, color=C[1], lw=2, marker="o", ms=4, label="full EM fit")
    x = np.array([df.n_nodes.min(), df.n_nodes.max()])
    ax.loglog(x, df.estep_s.iloc[-1] * x / x[-1], color=INK2, lw=0.8, ls="--", label="linear")
    ax.set_xlabel("Tree nodes")
    ax.set_ylabel("Seconds (one core)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "scalability.png")


def fig_selection():
    df = pd.read_csv(RES / "model_selection.csv")
    df["gain"] = df.heldout_ll - df.heldout_ll_no_tree
    main = df[(df.bl_model == "mut") & (df.delta == 0.5)]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))
    g = main.groupby("K")
    axes[0].errorbar(
        g.heldout_ll.mean().index,
        g.heldout_ll.mean(),
        yerr=g.heldout_ll.std(),
        color=C[0],
        lw=2,
        marker="o",
        ms=4,
        capsize=2,
        label="tHMM (tree + emissions)",
    )
    axes[0].errorbar(
        g.heldout_ll_no_tree.mean().index,
        g.heldout_ll_no_tree.mean(),
        yerr=g.heldout_ll_no_tree.std(),
        color=C[1],
        lw=2,
        marker="o",
        ms=4,
        capsize=2,
        label="same emissions, no tree",
    )
    axes[0].set_title("Held-out log-likelihood (20% of leaves)", fontsize=9)
    axes[0].legend(fontsize=7.5)
    axes[1].errorbar(
        g.gain.mean().index, g.gain.mean(), yerr=g.gain.std(), color=C[0], lw=2, marker="o", ms=4, capsize=2
    )
    axes[1].set_title("Lineage information gain (nats)", fontsize=9)
    for ax in axes[:2]:
        ax.set_xlabel("Hidden states K")
    sens = df[df.K.isin([6, 8, 10])].copy()
    sens["model"] = np.where(sens.bl_model == "unit", "unit", "n_mut + " + sens.delta.astype(str))
    order = ["n_mut + 0.1", "n_mut + 0.25", "n_mut + 0.5", "n_mut + 1.0", "n_mut + 2.0", "unit"]
    for i, K in enumerate([6, 8, 10]):
        s_ = sens[sens.K == K].groupby("model").heldout_ll.mean().reindex(order)
        axes[2].plot(range(len(order)), s_.values - s_.max(), color=C[i], lw=2, marker="o", ms=4, label=f"K={K}")
    axes[2].set_xticks(range(len(order)), order, rotation=30, ha="right")
    axes[2].set_ylabel("held-out LL − best")
    axes[2].set_title("Branch-length model", fontsize=9)
    axes[2].legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "model_selection.png")


def cluster_names():
    with open(RES / "kptracer_inputs.pkl", "rb") as f:
        recs = pickle.load(f)
    lab = np.concatenate([r["leiden"] for r in recs])
    name = np.concatenate([r["cluster_name"] for r in recs])
    return {int(k): str(v) for k, v in zip(lab, name, strict=True)}


def fig_transitions(name="anchored"):
    names = cluster_names()
    with open(RES / f"fit_{name}.pkl", "rb") as f:
        fit = pickle.load(f)
    labels = [names[int(k)] for k in fit["state_labels"]]
    thmm = pd.read_csv(RES / f"transitions_thmm_{name}.csv", index_col=0).values
    fitch = pd.read_csv(RES / f"transitions_fitch_{name}.csv", index_col=0).values
    Q = fit["Q"]
    rates = Q.copy()
    np.fill_diagonal(rates, np.nan)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, M, title in [
        (axes[0], rates, "tHMM rates Q (log10)"),
        (axes[1], thmm, "tHMM E[edge changes] (log10)"),
        (axes[2], fitch, "Fitch changes (log10)"),
    ]:
        M = M.astype(float).copy()
        np.fill_diagonal(M, np.nan)
        im = ax.imshow(np.log10(M + (1e-3 if ax is axes[0] else 0.5)), cmap="Blues")
        ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.set_xlabel("to")
        ax.set_ylabel("from")
        ax.set_title(title, fontsize=9)
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.7, label="log10")
    fig.tight_layout()
    fig.savefig(FIG / f"transitions_{name}.png")


def fig_benchmarks(name="anchored"):
    tb = pd.read_csv(RES / f"tumor_benchmarks_{name}.csv")
    from scipy.stats import spearmanr

    pairs = [
        ("fitch_plasticity", "Fitch plasticity (changes / edge)"),
        ("published_mean_scPlasticity", "Published mean scPlasticity"),
        ("path_autocorr_z_leiden", "PATH-style auto-correlation z (Leiden)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, (col, lab) in zip(axes, pairs, strict=True):
        ax.scatter(
            tb[col], tb.thmm_plasticity, s=np.clip(tb.n_cells / 20, 12, 120), color=C[0], edgecolor="white", lw=1
        )
        rho = spearmanr(tb[col], tb.thmm_plasticity)[0]
        ax.set_xlabel(lab)
        ax.set_ylabel("tHMM plasticity (E[changes] / edge)")
        ax.set_title(f"Spearman ρ = {rho:.2f} (21 tumors)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / f"tumor_benchmarks_{name}.png")


def fig_unsup_contingency():
    names = cluster_names()
    with open(RES / "fit_unsup.pkl", "rb") as f:
        fit = pickle.load(f)
    with open(RES / "kptracer_inputs.pkl", "rb") as f:
        recs = pickle.load(f)
    recs = [r for r in recs if r["tumor"] in fit["tumors"]]
    lab = np.concatenate([r["leiden"] for r in recs])
    gam = np.vstack([g[r["tree"].is_leaf] for g, r in zip(fit["gammas"], recs, strict=True)])
    uniq = np.unique(lab)
    M = np.array([gam[lab == c].sum(axis=0) for c in uniq])
    M = M / M.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(1.2 + 0.45 * M.shape[1], 5))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(uniq)), [names[int(c)] for c in uniq], fontsize=7)
    ax.set_xticks(range(M.shape[1]), [f"S{k}" for k in range(M.shape[1])], fontsize=7)
    ax.set_xlabel("tHMM hidden state")
    ax.set_title("Posterior state mass per published cluster", fontsize=9)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(FIG / "unsup_vs_clusters.png")


if __name__ == "__main__":
    import sys

    FIG.mkdir(exist_ok=True)
    which = sys.argv[1:] or ["simulation", "scalability", "selection", "transitions", "benchmarks", "unsup"]
    for w in which:
        {
            "simulation": fig_simulation,
            "scalability": fig_scalability,
            "selection": fig_selection,
            "transitions": fig_transitions,
            "benchmarks": fig_benchmarks,
            "unsup": fig_unsup_contingency,
        }[w]()
