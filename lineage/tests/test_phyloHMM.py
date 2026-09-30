"""Tests for the branch-length tHMM on reconstructed phylogenies."""

import itertools

import numpy as np
import pytest
import scipy.stats as sp
from scipy.integrate import quad_vec
from scipy.linalg import expm

from .. import ctmc
from ..HMM.E_step import get_beta_and_NF, get_gamma, get_MSD
from ..HMM.M_step import get_all_zetas, get_edge_posteriors
from ..LineageTree import LineageTree, get_scaled_Emission_Likelihoods
from ..phylo_sim import simulate_dataset
from ..phyloHMM import PhyloHMM, mask_leaves
from ..states.StateDistributionGaussian import StateDistribution as Gaussian
from ..tree_io import (
    align_observations,
    graph_to_phylotree,
    lca_character_states,
    load_tree,
    mutation_branch_lengths,
    newick_to_graph,
    to_lineage,
)

NEWICK = "((a:1,b:2)x:0.5,(c:1,(d:1)u:2,e:0.5):1)r;"


def small_lineage(D=2, rng=0):
    """Root with a binary and a ternary child subtree (one unifurcation collapsed); leaves observed."""
    rng = np.random.default_rng(rng)
    pt = load_tree(NEWICK)
    obs = rng.normal(size=(5, D))
    return pt, to_lineage(pt, ["a", "b", "c", "d", "e"], obs, [Gaussian(dim=D)])


def brute_force(model: PhyloHMM, lin: LineageTree):
    """Log-likelihood, node marginals and edge joints by enumerating every state assignment."""
    K, N = model.K, len(lin)
    T = model.edge_matrices(lin)
    logEL = np.column_stack([e.logpdf(lin.obs) for e in model.E])
    parents, children = lin.edges
    logw, marg, joint = [], np.zeros((N, K)), np.zeros((len(children), K, K))
    for z in itertools.product(range(K), repeat=N):
        z = np.array(z)
        lw = np.log(model.pi[z[0]]) + np.sum(np.log(T[children, z[parents], z[children]]))
        lw += np.sum(logEL[np.arange(N), z])
        logw.append((lw, z))
    m = max(lw for lw, _ in logw)
    total = sum(np.exp(lw - m) for lw, _ in logw)
    for lw, z in logw:
        p = np.exp(lw - m) / total
        marg[np.arange(N), z] += p
        joint[np.arange(len(children)), z[parents], z[children]] += p
    return m + np.log(total), marg, joint


# ---------------------------------------------------------------------------- emissions


def test_gaussian_logpdf_and_missing():
    g = Gaussian(mean=np.array([0.5, -1.0]), var=np.array([2.0, 0.5]))
    x = np.array([[0.1, 0.2], [np.nan, 0.3], [np.nan, np.nan]])
    ll = g.logpdf(x)
    ref = sp.norm.logpdf(0.1, 0.5, np.sqrt(2)) + sp.norm.logpdf(0.2, -1, np.sqrt(0.5))
    assert ll[0] == pytest.approx(ref)
    assert ll[1] == pytest.approx(sp.norm.logpdf(0.3, -1, np.sqrt(0.5)))
    assert ll[2] == 0.0


def test_gaussian_estimator_ignores_unobserved():
    rng = np.random.default_rng(0)
    x = rng.normal([3.0, -2.0], [1.0, 0.5], size=(4000, 2))
    x = np.vstack([x, np.full((100, 2), np.nan)])
    g = Gaussian(dim=2, ridge=0.0)
    g.estimator(x, np.ones(x.shape[0]))
    np.testing.assert_allclose(g.mean, [3.0, -2.0], atol=0.05)
    np.testing.assert_allclose(g.var, [1.0, 0.25], rtol=0.1)


def test_scaled_emissions_do_not_underflow():
    """With D = 300 the raw likelihoods are exp(-400) or smaller; the scaled ones stay finite."""
    rng = np.random.default_rng(1)
    pt, lin = small_lineage(D=300)
    lin.obs[pt.is_leaf] = rng.normal(0, 3, size=(5, 300))
    E = [Gaussian(dim=300), Gaussian(mean=np.ones(300))]
    EL, off = get_scaled_Emission_Likelihoods([lin], E)
    assert np.all(np.isfinite(EL[0])) and np.all(EL[0].max(axis=1) == 1.0)
    assert np.all(off[0][~pt.is_leaf] == 0.0)
    assert np.min(off[0][pt.is_leaf]) < -700


# ---------------------------------------------------------------------------- E-step


def test_per_edge_matches_shared_T():
    """Passing the same T per edge gives identical MSD, beta, gamma and zetas as one shared T."""
    rng = np.random.default_rng(2)
    _, lin = small_lineage()
    K = 3
    T = rng.dirichlet(np.ones(K), size=K)
    pi = rng.dirichlet(np.ones(K))
    EL = rng.random((len(lin), K))
    Tn = np.broadcast_to(T, (len(lin), K, K)).copy()
    res = []
    for TT in (T, Tn):
        MSD = get_MSD(lin.tree, pi, TT)
        NF, beta = get_beta_and_NF(lin.leaves_idx, lin.tree, TT, MSD, EL)
        gamma = get_gamma(lin.tree, TT, MSD, beta)
        res.append((MSD, NF, beta, gamma))
    for a, b in zip(*res, strict=True):
        np.testing.assert_allclose(a, b)
    MSD, _, beta, gamma = res[0]
    _, _, xi = get_edge_posteriors(lin.tree, beta, MSD, gamma, Tn)
    np.testing.assert_allclose(xi.sum(axis=0), get_all_zetas(lin.tree, beta, MSD, gamma, T))


@pytest.mark.parametrize("mode", ["ctmc", "discrete"])
def test_estep_matches_enumeration(mode):
    """Log-likelihood, marginals and edge joints agree with brute-force enumeration on a multifurcating tree."""
    rng = np.random.default_rng(3)
    _, lin = small_lineage(D=2)
    m = PhyloHMM([lin], 2, mode=mode, E=Gaussian(dim=2), rng=4)
    m.E = [Gaussian(mean=rng.normal(size=2)), Gaussian(mean=rng.normal(size=2), var=np.array([0.5, 2.0]))]
    m.pi = np.array([0.3, 0.7])
    m.Q = ctmc.rates_to_Q(np.array([[0, 0.4], [0.9, 0]]))
    post = m.e_step()[0]
    LL, marg, joint = brute_force(m, lin)
    assert post.LL == pytest.approx(LL)
    np.testing.assert_allclose(post.gamma, marg, atol=1e-10)
    np.testing.assert_allclose(post.xi, joint, atol=1e-10)


# ---------------------------------------------------------------------------- CTMC


def test_ctmc_statistics_match_quadrature():
    rng = np.random.default_rng(5)
    K = 3
    Q = ctmc.random_Q(K, 1.2, rng)
    t = np.array([0.4, 1.3])
    xi = rng.dirichlet(np.ones(K * K), size=2).reshape(2, K, K)
    dwell, jumps = ctmc.expected_statistics(Q, t, xi)
    ref_d, ref_n = np.zeros(K), np.zeros((K, K))
    for e in range(2):
        P = expm(Q * t[e])
        for i, j in itertools.product(range(K), repeat=2):
            integrand = lambda s, i=i, j=j, e=e: np.outer(expm(Q * s)[:, i], expm(Q * (t[e] - s))[j, :])  # noqa: E731
            val = np.sum(xi[e] * quad_vec(integrand, 0, t[e])[0] / P)
            if i == j:
                ref_d[i] += val
            else:
                ref_n[i, j] += Q[i, j] * val
    np.testing.assert_allclose(dwell, ref_d, atol=1e-10)
    np.testing.assert_allclose(jumps, ref_n, atol=1e-10)
    assert dwell.sum() == pytest.approx(t.sum())
    np.testing.assert_allclose(ctmc.transition_matrices(Q, t)[1], expm(Q * 1.3), atol=1e-12)


def test_ctmc_repeated_eigenvalues():
    Q = ctmc.rates_to_Q(np.ones((3, 3)))
    np.testing.assert_allclose(ctmc.transition_matrices(Q, np.array([0.7]))[0], expm(Q * 0.7), atol=1e-8)
    dwell, _ = ctmc.expected_statistics(Q, np.array([0.7]), np.full((1, 3, 3), 1 / 9))
    assert dwell.sum() == pytest.approx(0.7)


def test_stationary():
    Q = ctmc.random_Q(4, 1.0, 6)
    pi = ctmc.stationary(Q)
    np.testing.assert_allclose(pi @ Q, 0, atol=1e-12)


# ---------------------------------------------------------------------------- tree loading


def test_newick_loader_prunes_and_collapses():
    pt = load_tree(NEWICK)
    assert pt.names[0] == "r"
    # d's unifurcating parent u is merged: d hangs off the ternary node with length 2 + 1
    assert "u" not in set(pt.names)
    d = int(np.nonzero(pt.names == "d")[0][0])
    assert pt.branch_lengths[d] == pytest.approx(3.0)
    # Parents come before children, and the last node is a leaf
    par = pt.parents
    assert np.all(par[1:] < np.arange(1, len(par)))
    assert pt.is_leaf[-1]
    assert sorted(pt.leaf_names) == ["a", "b", "c", "d", "e"]

    pruned = load_tree(NEWICK, keep_leaves=["a", "c", "e"])
    assert sorted(pruned.leaf_names) == ["a", "c", "e"]
    a = int(np.nonzero(pruned.names == "a")[0][0])
    assert pruned.branch_lengths[a] == pytest.approx(1.5)  # x collapsed into a


def test_graph_roundtrip_and_unnamed_nodes():
    g = newick_to_graph("((a:1,b:1):1,c:2);")
    assert len(g) == 5
    pt = graph_to_phylotree(g)
    g2 = pt.to_graph()
    assert set(g2.edges) == set(g.edges)


def test_align_observations():
    pt = load_tree(NEWICK)
    obs = align_observations(pt, ["e", "a", "zz"], np.array([[5.0], [1.0], [9.0]]))
    assert obs[pt.names == "a"][0, 0] == 1.0
    assert obs[pt.names == "e"][0, 0] == 5.0
    assert np.isnan(obs[pt.names == "b"][0, 0])
    assert np.all(np.isnan(obs[~pt.is_leaf]))


def test_lca_characters_and_mutation_lengths():
    pt = load_tree("((a,b)x,c)r;")
    chars = np.zeros((len(pt.names), 3), dtype=int)
    rows = {"a": [1, 2, -1], "b": [1, 3, 4], "c": [0, 2, 4]}
    for n, v in rows.items():
        chars[pt.names == n] = v
    anc = lca_character_states(pt, chars)
    np.testing.assert_array_equal(anc[pt.names == "x"][0], [1, 0, 4])
    np.testing.assert_array_equal(anc[pt.names == "r"][0], [0, 0, 4])
    bl = mutation_branch_lengths(pt, anc)
    assert bl[pt.names == "x"][0] == 1  # site 0 cut
    assert bl[pt.names == "a"][0] == 1  # site 1 cut; site 2 missing
    assert bl[pt.names == "c"][0] == 1


# ---------------------------------------------------------------------------- fitting


def simulated(sample_frac=1.0, rng=0):
    rng = np.random.default_rng(rng)
    Q = ctmc.rates_to_Q(np.array([[0, 0.3, 0.05], [0.1, 0, 0.2], [0.02, 0.05, 0]]))
    pi = np.array([0.8, 0.15, 0.05])
    means = rng.normal(0, 1.5, (3, 4))
    data = simulate_dataset(8, 150, Q, pi, means, 1.0, sample_frac=sample_frac, rng=rng)
    X = [to_lineage(d["tree"], d["tree"].leaf_names, d["obs"][d["tree"].is_leaf], [Gaussian(dim=4)]) for d in data]
    return X, data, Q, means


def test_fit_monotone_and_recovers_Q():
    X, _, Q, means = simulated()
    m = PhyloHMM(X, 3, rng=1).fit(tol=1e-6)
    assert np.all(np.diff(m.LL_trace) > -1e-6), "EM must not decrease the likelihood"
    order = [int(np.argmin([np.linalg.norm(e.mean - mu) for e in m.E])) for mu in means]
    assert sorted(order) == [0, 1, 2]
    Qhat = m.Q[np.ix_(order, order)]
    np.testing.assert_allclose(Qhat, Q, atol=0.12)
    summ = m.switch_summary()
    assert all(0 <= s["plasticity"] <= 1 for s in summ)
    assert all(s["jumps"].sum() >= s["expected_changes"] - 1e-8 for s in summ)


def test_heldout_loglik():
    X, _, _, _ = simulated(rng=2)
    Xm = mask_leaves(X, 0.3, rng=0)
    m = PhyloHMM(Xm, 3, rng=0).fit(max_iter=30)
    ll = m.heldout_loglik(Xm, X)
    assert np.isfinite(ll) and ll < 0
    assert np.isfinite(m.BIC())


def test_discrete_mode_fits():
    X, _, _, _ = simulated(rng=3)
    for x in X:
        x.branch_lengths = None
    m = PhyloHMM(X, 3, mode="discrete", rng=0).fit(max_iter=30)
    np.testing.assert_allclose(m.T.sum(axis=1), 1.0)
    assert np.all(np.diff(m.LL_trace) > -1e-6)


# ---------------------------------------------------------------------------- baselines


def test_small_parsimony_multifurcation():
    from ..phylo_stats import small_parsimony

    pt = load_tree("((a,b,c)x,(d,e)y)r;")
    lab = np.full(len(pt.names), -1)
    for n, v in {"a": 0, "b": 0, "c": 1, "d": 1, "e": 1}.items():
        lab[pt.names == n] = v
    score, labels, counts = small_parsimony(pt, lab, 2)
    assert score == 2  # x is 0 (one change to c), and r -> y is one change
    assert counts.sum() == score
    assert labels[pt.names == "x"][0] == 0 and labels[pt.names == "y"][0] == 1


def test_phylogenetic_correlation_signs():
    from ..phylo_stats import leaf_node_distance, phylogenetic_correlation

    pt = load_tree("(((a,b),(c,d)),((e,f),(g,h)));")
    dist = leaf_node_distance(pt)
    clustered = np.array([0 if n in "abcd" else 1 for n in pt.leaf_names])
    mixed = np.array([0 if n in "aceg" else 1 for n in pt.leaf_names])
    assert phylogenetic_correlation(clustered, dist, 2)[0, 0] > 0
    assert phylogenetic_correlation(mixed, dist, 2)[0, 0] < phylogenetic_correlation(clustered, dist, 2)[0, 0]


def test_large_polytomy_is_stable():
    """A root with 5000 leaf children (as in saturated recorder trees) must not overflow the upward pass."""
    n = 5000
    newick = "(" + ",".join(f"c{i}:1" for i in range(n)) + ")r;"
    pt = load_tree(newick)
    rng = np.random.default_rng(0)
    lin = to_lineage(pt, pt.leaf_names, rng.normal(size=(n, 3)), [Gaussian(dim=3)])
    m = PhyloHMM([lin], 3, rng=0)
    m.E = [Gaussian(mean=np.full(3, v)) for v in (-1.0, 0.0, 1.0)]
    post = m.e_step()[0]
    assert np.isfinite(post.LL)
    np.testing.assert_allclose(post.gamma.sum(axis=1), 1.0)
    # With a single internal node the likelihood factorizes over the root state
    logEL = np.column_stack([e.logpdf(lin.obs[1:]) for e in m.E])
    T = ctmc.transition_matrices(m.Q, np.array([1.0]))[0]
    from scipy.special import logsumexp

    per_root = np.array([np.sum(logsumexp(np.log(T[k])[None, :] + logEL, axis=1)) for k in range(3)])
    assert post.LL == pytest.approx(logsumexp(per_root + np.log(m.pi)))
