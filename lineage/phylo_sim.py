"""Simulate branch-length trees with CTMC state changes and Gaussian leaf emissions."""

import networkx as nx
import numpy as np
from scipy.linalg import expm

from .tree_io import PhyloTree, graph_to_phylotree, prune_graph


def yule_tree(n_leaves: int, birth_rate: float = 1.0, rng=None) -> nx.DiGraph:
    """Pure-birth tree grown until ``n_leaves`` lineages exist; branch lengths are in time units.

    Every leaf is sampled at the same final time (the moment the last split happens plus an
    exponential waiting time), as in a tumor harvested at one time point.
    """
    rng = np.random.default_rng(rng)
    g = nx.DiGraph()
    g.add_node("n0", time=0.0)
    active = ["n0"]
    t = 0.0
    counter = 1
    while len(active) < n_leaves:
        t += rng.exponential(1.0 / (birth_rate * len(active)))
        parent = active.pop(int(rng.integers(len(active))))
        g.nodes[parent]["time_end"] = t
        for _ in range(2):
            child = f"n{counter}"
            counter += 1
            g.add_node(child, time=t)
            active.append(child)
            g.add_edge(parent, child)
    t_end = t + rng.exponential(1.0 / (birth_rate * len(active)))
    for n in active:
        g.nodes[n]["time_end"] = t_end
    # A node's edge length is the time from its parent's split to its own split (or sampling)
    for u, v in g.edges:
        g[u][v]["length"] = g.nodes[v]["time_end"] - g.nodes[v]["time"]
    # The root lineage lives from 0 until its split; fold that stem into nothing (root state is at its split)
    return g


def simulate_states(pt: PhyloTree, Q: np.ndarray, pi: np.ndarray, rng=None) -> np.ndarray:
    """Sample node states from the root distribution and exp(Q t) along each edge."""
    rng = np.random.default_rng(rng)
    K = Q.shape[0]
    states = np.zeros(len(pt.names), dtype=int)
    states[0] = rng.choice(K, p=pi)
    par = pt.parents
    cache: dict[float, np.ndarray] = {}
    for i in range(1, len(states)):
        t = float(pt.branch_lengths[i])
        if t not in cache:
            P = np.clip(expm(Q * t), 0, None)
            cache[t] = P / P.sum(axis=1, keepdims=True)
        states[i] = rng.choice(K, p=cache[t][states[par[i]]])
    return states


def simulate_dataset(
    n_trees: int,
    n_leaves: int,
    Q: np.ndarray,
    pi: np.ndarray,
    means: np.ndarray,
    sd: float | np.ndarray,
    sample_frac: float = 1.0,
    rng=None,
) -> list[dict]:
    """Simulate trees, states and leaf observations, then subsample leaves.

    Subsampled trees are pruned to the retained leaves with unifurcations collapsed (branch lengths
    summed), as for a real recorder experiment that sequences a fraction of the tumor.

    :return: one dict per tree with the full and sampled PhyloTrees, the true states of the sampled
        tree's nodes (internal nodes of the pruned tree are real ancestors), and leaf observations.
    """
    rng = np.random.default_rng(rng)
    K, D = means.shape
    sd = np.broadcast_to(np.asarray(sd, dtype=float), (K, D))
    out = []
    for _ in range(n_trees):
        g = yule_tree(n_leaves, rng=rng)
        full = graph_to_phylotree(g)
        states = simulate_states(full, Q, pi, rng)
        state_of = dict(zip(full.names, states, strict=True))
        leaves = full.leaf_names
        keep = leaves[rng.random(leaves.size) < sample_frac] if sample_frac < 1 else leaves
        if keep.size < 3:
            keep = rng.choice(leaves, size=3, replace=False)
        sampled = graph_to_phylotree(prune_graph(g, keep, collapse_unifurcations=True))
        s_states = np.array([state_of[n] for n in sampled.names])
        obs = np.full((len(sampled.names), D), np.nan)
        leaf = sampled.is_leaf
        obs[leaf] = rng.normal(means[s_states[leaf]], sd[s_states[leaf]])
        out.append({"full": full, "tree": sampled, "states": s_states, "obs": obs})
    return out
