"""Baselines that label the leaves first and then measure how the labels relate to the tree.

* :func:`small_parsimony` is unit-cost Sankoff parsimony, which equals Fitch-Hartigan on
  multifurcating trees and is what Cassiopeia's FitchCount builds on.
* :func:`phylogenetic_correlation` is the Moran's-I phylogenetic auto/cross-correlation used by
  PATH (Schiffman et al. 2024), with inverse node-distance weights.
"""

import numpy as np
from scipy.sparse.csgraph import shortest_path

from .tree_io import PhyloTree


def small_parsimony(pt: PhyloTree, leaf_labels: np.ndarray, K: int) -> tuple[int, np.ndarray, np.ndarray]:
    """Minimum number of label changes, one optimal labeling of every node, and its transition counts.

    :param leaf_labels: integer label per node (only leaves are read); -1 marks an unlabeled leaf.
    :return: parsimony score, per-node labels, and a K by K matrix counting parent -> child changes.
    """
    N = len(pt.names)
    tree = pt.tree
    cost = np.zeros((N, K))
    leaf = pt.is_leaf
    for i in np.nonzero(leaf)[0]:
        if leaf_labels[i] >= 0:
            cost[i] = np.inf
            cost[i, leaf_labels[i]] = 0.0
    change = 1.0 - np.eye(K)
    for p in np.nonzero(~leaf)[0][::-1]:
        ch = tree.indices[tree.indptr[p] : tree.indptr[p + 1]]
        # min over the child's state of (child cost + change cost), for each parent state
        cost[p] = np.sum(np.min(cost[ch][:, np.newaxis, :] + change[np.newaxis], axis=2), axis=0)

    labels = np.full(N, -1)
    labels[0] = int(np.argmin(cost[0]))
    counts = np.zeros((K, K), dtype=int)
    for p in np.nonzero(~leaf)[0]:
        for c in tree.indices[tree.indptr[p] : tree.indptr[p + 1]]:
            opt = cost[c] + change[labels[p]]
            # Prefer keeping the parent's label among ties, as Fitch does
            best = labels[p] if opt[labels[p]] <= opt.min() else int(np.argmin(opt))
            labels[c] = best
            counts[labels[p], best] += 1
    np.fill_diagonal(counts, 0)
    return int(counts.sum()), labels, counts


def leaf_node_distance(pt: PhyloTree, leaves: np.ndarray | None = None, weighted: bool = False) -> np.ndarray:
    """Pairwise path lengths between leaves, in edges (or branch lengths if ``weighted``)."""
    leaves = np.nonzero(pt.is_leaf)[0] if leaves is None else leaves
    A = pt.tree.astype(float)
    if weighted:
        A = A.multiply(pt.branch_lengths[np.newaxis, :]).tocsr()
    return shortest_path(A, directed=False, indices=leaves)[:, leaves]


def _inverse_distance_weights(dist: np.ndarray) -> np.ndarray:
    W = np.zeros_like(dist, dtype=float)
    off = dist > 0
    W[off] = 1.0 / dist[off]
    return W / W.sum()


def _moran(W: np.ndarray, labels: np.ndarray, K: int) -> np.ndarray:
    onehot = np.eye(K)[labels]
    Z = onehot - onehot.mean(axis=0)
    norm = np.sqrt(np.sum(Z**2, axis=0))
    norm[norm == 0] = np.inf
    return labels.size * (Z.T @ W @ Z) / np.outer(norm, norm)


def phylogenetic_correlation(labels: np.ndarray, dist: np.ndarray, K: int) -> np.ndarray:
    """PATH-style Moran's I between one-hot state indicators, with weights :math:`w_{ij} = 1/d_{ij}`.

    Diagonal entries are the phylogenetic auto-correlations (heritability of each state); off-diagonal
    entries are cross-correlations (positive means the two states sit in nearby clades).
    """
    return _moran(_inverse_distance_weights(dist), labels, K)


def phylogenetic_correlation_z(labels: np.ndarray, dist: np.ndarray, K: int, n_perm: int = 100, rng=None):
    """Moran's I (as :func:`phylogenetic_correlation`) and its z-score against leaf-label permutations."""
    rng = np.random.default_rng(rng)
    W = _inverse_distance_weights(dist)
    obs = _moran(W, labels, K)
    perms = np.array([_moran(W, rng.permutation(labels), K) for _ in range(n_perm)])
    sd = perms.std(axis=0)
    sd[sd == 0] = np.inf
    return obs, (obs - perms.mean(axis=0)) / sd
