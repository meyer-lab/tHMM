"""Build per-tumor tree + embedding inputs from the KP-Tracer Zenodo release (record 5847462).

Run with ``uv run --with anndata python analysis/kptracer/prepare_data.py [DATA_DIR]``; DATA_DIR defaults
to ``$KPTRACER_DATA`` or ``~/data/kptracer/KPTracer-Data``. Writes ``results/kptracer_inputs.pkl``.

Filtering follows Yang et al. 2022 (Figure 4 notebook in KPTracer-release): sgNT primary tumors with a
tree, at least 100 cells, >5% unique alleles and >20% unsaturated targets; leaves restricted to cells in
the expression AnnData; clusters with <=2.5% of a tumor's cells dropped; unifurcations collapsed.
"""

import os
import pickle
import sys
from pathlib import Path
from typing import Any

import anndata as ad  # ty: ignore[unresolved-import]  (only needed here; run with `uv run --with anndata`)
import numpy as np
import pandas as pd

from lineage.tree_io import (
    graph_to_phylotree,
    lca_character_states,
    mutation_branch_lengths,
    newick_to_graph,
    prune_graph,
)

OUT = Path(__file__).parent / "results"
FILTER_PROP = 0.025


def data_dir() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    return Path(os.environ.get("KPTRACER_DATA", Path.home() / "data/kptracer/KPTracer-Data"))


def load_character_matrix(path: Path) -> pd.DataFrame:
    cm = pd.read_csv(path, sep="\t", index_col=0, dtype=str)
    return cm.replace("-", "-1").astype(int)


def add_pca(adata, records, n_genes=2000, n_pcs=10):
    """Sensitivity embedding: PCA of the log-normalized expression (not batch corrected), same dimension as scVI."""
    from sklearn.decomposition import PCA

    names = np.concatenate([r["leaf_names"] for r in records])
    rows = np.sort(adata.obs_names.get_indexer(names))
    X = adata.X[rows]
    gene_var = np.asarray(X.power(2).mean(axis=0)).ravel() - np.asarray(X.mean(axis=0)).ravel() ** 2
    genes = np.argsort(gene_var)[::-1][:n_genes]
    pcs = PCA(n_pcs, random_state=0).fit_transform(X[:, genes].toarray())
    lookup = pd.DataFrame(pcs, index=adata.obs_names[rows])
    for r in records:
        r["pca"] = lookup.loc[r["leaf_names"]].values


def main():
    root = data_dir()
    adata = ad.read_h5ad(root / "expression/adata_processed.nt.h5ad", backed="r")
    obs = adata.obs[["Tumor", "leiden_sub", "Cluster-Name"]].copy()
    scvi = pd.DataFrame(np.asarray(adata.obsm["X_scVI"]), index=adata.obs_names)
    stats = pd.read_csv(root / "tumor_statistics.tsv", sep="\t", index_col=0)
    published = pd.read_csv(root / "plasticity_scores.tsv", sep="\t", index_col=0)

    passing = stats[(stats.NumCells >= 100) & (stats.PercentUnique > 0.05) & (stats.PercentUnsaturatedTargets > 0.2)]
    tumors = sorted(
        t
        for t in passing.index
        if t.split("_")[1] == "NT"
        and t.split("_")[2].startswith("T")
        and len(t.split("_")) == 3
        and (root / f"trees/{t}_tree.nwk").exists()
    )

    records: list[dict[str, Any]] = []
    for tumor in tumors:
        g = newick_to_graph(root / f"trees/{tumor}_tree.nwk")
        leaves = [n for n in g if g.out_degree(n) == 0]
        cells = obs.index.intersection(leaves)
        cells = cells[obs.loc[cells, "Tumor"].astype(str).values == tumor]
        counts = obs.loc[cells, "leiden_sub"].value_counts()
        keep_clusters = counts.index[counts > FILTER_PROP * len(cells)]
        cells = cells[obs.loc[cells, "leiden_sub"].isin(keep_clusters).values]
        if len(cells) < 20:
            continue
        pt = graph_to_phylotree(prune_graph(g, cells, collapse_unifurcations=True))

        cm = load_character_matrix(root / f"trees/{tumor}_character_matrix.txt")
        chars = np.full((len(pt.names), cm.shape[1]), -1, dtype=int)
        leaf_idx = np.nonzero(pt.is_leaf)[0]
        chars[leaf_idx] = cm.loc[list(pt.names[leaf_idx])].values
        anc = lca_character_states(pt, chars)
        n_mut = mutation_branch_lengths(pt, anc)

        leaf_names = pt.leaf_names.astype(str)
        records.append(
            {
                "tumor": tumor,
                "tree": pt,
                "n_mut": n_mut,
                "leaf_names": leaf_names,
                "scvi": scvi.loc[leaf_names].values,
                "leiden": obs.loc[leaf_names, "leiden_sub"].astype(int).values,
                "cluster_name": obs.loc[leaf_names, "Cluster-Name"].astype(str).values,
                "published_scPlasticity": published.reindex(leaf_names)["scPlasticity"].values,
            }
        )
        print(tumor, len(leaf_names), len(pt.names), f"zero-mutation edges {np.mean(n_mut[1:] == 0):.2f}", flush=True)

    add_pca(adata, records)
    OUT.mkdir(exist_ok=True)
    with open(OUT / "kptracer_inputs.pkl", "wb") as f:
        pickle.dump(records, f)
    print(len(records), "tumors,", sum(len(r["leaf_names"]) for r in records), "cells")


if __name__ == "__main__":
    main()
