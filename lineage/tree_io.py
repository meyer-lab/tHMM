"""Load reconstructed phylogenies (Newick, networkx, or Cassiopeia) into the CSR form used by LineageTree."""

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import networkx as nx
import numpy as np
from Bio import Phylo
from scipy.sparse import csr_array

from .LineageTree import LineageTree


@dataclass
class PhyloTree:
    """A rooted tree in breadth-first order: the root is index 0 and parents precede their children."""

    tree: csr_array
    names: np.ndarray
    branch_lengths: np.ndarray  # length of the edge into each node; 0 for the root

    @property
    def is_leaf(self) -> np.ndarray:
        return np.diff(self.tree.indptr) == 0

    @property
    def leaf_names(self) -> np.ndarray:
        return self.names[self.is_leaf]

    @property
    def parents(self) -> np.ndarray:
        """Parent index of each node (-1 for the root)."""
        par = np.full(len(self.names), -1)
        par[self.tree.indices] = np.repeat(np.arange(len(self.names)), np.diff(self.tree.indptr))
        return par

    def depth(self) -> np.ndarray:
        """Sum of branch lengths from the root to each node."""
        d = np.zeros(len(self.names))
        par = self.parents
        for i in range(1, len(d)):  # BFS order, so the parent is already filled in
            d[i] = d[par[i]] + self.branch_lengths[i]
        return d

    def to_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_nodes_from(self.names)
        par = self.parents
        for i in range(1, len(self.names)):
            g.add_edge(self.names[par[i]], self.names[i], length=float(self.branch_lengths[i]))
        return g


def newick_to_graph(newick: str | Path) -> nx.DiGraph:
    """Parse a Newick file or string into a DiGraph with a ``length`` attribute on each edge.

    Unnamed internal nodes are named ``node{i}``. Missing branch lengths are recorded as NaN.
    """
    text = Path(newick).read_text() if _is_file(newick) else str(newick)
    phylo = Phylo.read(StringIO(text.strip()), "newick")
    g = nx.DiGraph()
    counter = 0

    def name_of(clade) -> str:
        nonlocal counter
        if clade.name is None or clade.name == "":
            clade.name = f"node{counter}"
            counter += 1
        return clade.name

    root_name = name_of(phylo.root)
    g.add_node(root_name)
    stack = [phylo.root]
    while stack:
        clade = stack.pop()
        for child in clade.clades:
            bl = np.nan if child.branch_length is None else float(child.branch_length)
            g.add_edge(clade.name, name_of(child), length=bl)
            stack.append(child)
    assert nx.is_arborescence(g), "The Newick string does not describe a rooted tree with unique node names."
    return g


def _is_file(x) -> bool:
    if isinstance(x, Path):
        return True
    try:
        return Path(str(x)).is_file()
    except OSError:  # a long Newick string is not a valid path
        return False


def cassiopeia_to_graph(ctree) -> nx.DiGraph:
    """Convert a ``cassiopeia.data.CassiopeiaTree`` without importing Cassiopeia."""
    g = nx.DiGraph()
    for u, v in ctree.edges:
        g.add_edge(u, v, length=float(ctree.get_branch_length(u, v)))
    return g


def prune_graph(g: nx.DiGraph, keep_leaves: Iterable[str], collapse_unifurcations: bool = True) -> nx.DiGraph:
    """Keep only the given leaves (and their ancestors), then merge single-child nodes into their child.

    Branch lengths are summed through collapsed nodes. The root is kept even if it has one child, unless
    that child is the only path to the retained leaves and the root itself would then be a unifurcation.
    """
    g = g.copy()
    keep = set(keep_leaves)
    leaves = [n for n in g if g.out_degree(n) == 0]
    g.remove_nodes_from([n for n in leaves if n not in keep])
    # Remove internal nodes left without descendants
    while True:
        dead = [n for n in g if g.out_degree(n) == 0 and n not in keep]
        if not dead:
            break
        g.remove_nodes_from(dead)

    if collapse_unifurcations:
        root = next(n for n in g if g.in_degree(n) == 0)
        # Collapse a unifurcating root into its child
        while g.out_degree(root) == 1:
            (child,) = g.successors(root)
            g.remove_node(root)
            root = child
        for n in list(nx.dfs_postorder_nodes(g, root)):
            if n != root and g.out_degree(n) == 1:
                (parent,) = g.predecessors(n)
                (child,) = g.successors(n)
                length = _length(g, parent, n) + _length(g, n, child)
                g.remove_node(n)
                g.add_edge(parent, child, length=length)
    return g


def _length(g: nx.DiGraph, u, v) -> float:
    return float(g[u][v].get("length", np.nan))


def graph_to_phylotree(g: nx.DiGraph) -> PhyloTree:
    """Order nodes breadth-first from the root and build the parent-to-child CSR adjacency."""
    roots = [n for n in g if g.in_degree(n) == 0]
    assert len(roots) == 1, "The graph must have exactly one root."
    order = []
    q = deque(roots)
    while q:
        n = q.popleft()
        order.append(n)
        q.extend(sorted(g.successors(n), key=str))
    idx = {n: i for i, n in enumerate(order)}
    N = len(order)
    indptr = np.zeros(N + 1, dtype=np.int32)
    indices = []
    bl = np.zeros(N)
    for i, n in enumerate(order):
        ch = [idx[c] for c in sorted(g.successors(n), key=str)]
        indices.extend(ch)
        indptr[i + 1] = indptr[i] + len(ch)
        for c in g.successors(n):
            bl[idx[c]] = _length(g, n, c)
    tree = csr_array((np.ones(len(indices), dtype=bool), np.array(indices, dtype=np.int32), indptr), shape=(N, N))
    return PhyloTree(tree=tree, names=np.array(order, dtype=object), branch_lengths=bl)


def load_tree(source, keep_leaves: Iterable[str] | None = None, collapse_unifurcations: bool = True) -> PhyloTree:
    """Load a tree from a Newick path/string, a networkx DiGraph, or a CassiopeiaTree.

    :param keep_leaves: If given, prune to these leaves (e.g. those with expression data).
    """
    if isinstance(source, nx.DiGraph):
        g = source
    elif hasattr(source, "get_branch_length") and hasattr(source, "edges"):
        g = cassiopeia_to_graph(source)
    else:
        g = newick_to_graph(source)
    if keep_leaves is not None or collapse_unifurcations:
        if keep_leaves is None:
            keep_leaves = [n for n in g if g.out_degree(n) == 0]
        g = prune_graph(g, keep_leaves, collapse_unifurcations)
    return graph_to_phylotree(g)


def lca_character_states(pt: PhyloTree, leaf_characters: np.ndarray, missing: int = -1) -> np.ndarray:
    """Cassiopeia-style ancestral reconstruction: a node keeps a character state only if all children
    that observe it agree; otherwise it is uncut (0). Missing data propagates only if all children miss it.

    :param leaf_characters: (N by C) integer matrix; rows for internal nodes are ignored.
    """
    chars = np.array(leaf_characters, dtype=int, copy=True)
    for p in np.nonzero(~pt.is_leaf)[0][::-1]:
        ch = pt.tree.indices[pt.tree.indptr[p] : pt.tree.indptr[p + 1]]
        sub = chars[ch]
        state = np.zeros(sub.shape[1], dtype=int)
        for c in range(sub.shape[1]):
            vals = sub[:, c][sub[:, c] != missing]
            if vals.size == 0:
                state[c] = missing
            elif np.all(vals == vals[0]):
                state[c] = vals[0]
        chars[p] = state
    return chars


def mutation_branch_lengths(pt: PhyloTree, node_characters: np.ndarray, missing: int = -1) -> np.ndarray:
    """Number of characters that change along the edge into each node (0 for the root)."""
    par = pt.parents
    bl = np.zeros(len(pt.names))
    c = node_characters[1:]
    p = node_characters[par[1:]]
    present = (c != missing) & (p != missing)
    bl[1:] = np.sum(present & (c != p), axis=1)
    return bl


def align_observations(pt: PhyloTree, obs_names: Sequence[str] | np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Place per-cell observations at the matching leaves; internal nodes and unmatched leaves get NaN."""
    obs = np.asarray(obs, dtype=float)
    lookup = {str(n): i for i, n in enumerate(obs_names)}
    out = np.full((len(pt.names), obs.shape[1]), np.nan)
    for i, name in enumerate(pt.names):
        if pt.is_leaf[i] and str(name) in lookup:
            out[i] = obs[lookup[str(name)]]
    return out


def to_lineage(
    pt: PhyloTree,
    obs_names: Sequence[str] | np.ndarray,
    obs: np.ndarray,
    E: Sequence,
    branch_lengths: np.ndarray | None = None,
) -> LineageTree:
    """Build a LineageTree with NaN observations at internal nodes and the tree's branch lengths."""
    bl = pt.branch_lengths if branch_lengths is None else branch_lengths
    lin = LineageTree(pt.tree, E, obs=align_observations(pt, obs_names, obs), branch_lengths=np.asarray(bl, float))
    lin.names = pt.names
    return lin
