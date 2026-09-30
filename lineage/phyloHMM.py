r"""Latent-state tHMM on reconstructed phylogenies with branch-length-dependent transitions.

Each edge e into node c has transition matrix :math:`T_e = \exp(Q t_e)` (``mode="ctmc"``), or a single
shared matrix T for division-resolved trees (``mode="discrete"``). Emissions are only observed at the
leaves; internal nodes carry NaN observations and emission likelihood 1. The upward-downward passes
reuse :mod:`lineage.HMM.E_step` with per-edge matrices.
"""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans

from . import ctmc
from .HMM.E_step import get_beta_and_logNF, get_gamma, get_MSD
from .HMM.M_step import get_edge_posteriors
from .LineageTree import LineageTree, get_scaled_Emission_Likelihoods


@dataclass
class TreePosterior:
    """E-step output for one tree."""

    gamma: np.ndarray  # (N, K) marginal posteriors P(z_n = k | Y)
    parents: np.ndarray  # (E,)
    children: np.ndarray  # (E,)
    xi: np.ndarray  # (E, K, K) joint posteriors P(z_p = k, z_c = l | Y)
    LL: float


def _lengths(lin: LineageTree) -> np.ndarray:
    assert lin.branch_lengths is not None, "ctmc mode needs branch lengths."
    return lin.branch_lengths


class PhyloHMM:
    """Hidden Markov model on a collection of rooted trees with leaf-only observations."""

    def __init__(
        self,
        X: list[LineageTree],
        num_states: int,
        mode: str = "ctmc",
        E=None,
        rng=None,
        fix_Q: np.ndarray | None = None,
        fix_E=None,
        pi_mode: str = "roots",
    ):
        """
        :param X: Trees. In ``ctmc`` mode each must have ``branch_lengths``.
        :param num_states: Number of hidden states K.
        :param mode: ``"ctmc"`` for :math:`\\exp(Q t_e)` per edge, or ``"discrete"`` for one T per edge.
        :param E: Template emission object copied for each state (ignored if ``fix_E`` is given).
        :param pi_mode: ``"roots"`` estimates the root distribution from the root posteriors;
            ``"stationary"`` ties it to the stationary distribution of Q or T.
        """
        assert mode in ("ctmc", "discrete")
        assert pi_mode in ("roots", "stationary")
        self.X = X
        self.K = num_states
        self.mode = mode
        self.pi_mode = pi_mode
        self.rng = np.random.default_rng(rng)
        if mode == "ctmc":
            assert all(x.branch_lengths is not None for x in X), "ctmc mode needs branch lengths."
        self.fix_Q = fix_Q
        self.fix_E = fix_E
        template = E if E is not None else X[0].E[0]
        self.E = deepcopy(fix_E) if fix_E is not None else [deepcopy(template) for _ in range(num_states)]
        self.pi = np.ones(num_states) / num_states
        scale = self._typical_length()
        self.Q = fix_Q.copy() if fix_Q is not None else ctmc.random_Q(num_states, 0.5 / scale, self.rng)
        self.T = self.rng.dirichlet(np.ones(num_states) * 5, size=num_states) * 0.5 + 0.5 * np.eye(num_states)
        self.LL_trace: list[float] = []

    # ---------------------------------------------------------------- helpers
    def _typical_length(self) -> float:
        if self.mode != "ctmc":
            return 1.0
        bl = np.concatenate([_lengths(x)[1:] for x in self.X])
        bl = bl[bl > 0]
        return float(np.median(bl)) if bl.size else 1.0

    def edge_matrices(self, lin: LineageTree) -> np.ndarray:
        """Per-node transition matrices (N, K, K); entry 0 (the root) is unused."""
        if self.mode == "discrete":
            return np.broadcast_to(self.T, (len(lin), self.K, self.K))
        return ctmc.transition_matrices(self.Q, _lengths(lin))

    def observed_leaf_mask(self, lin: LineageTree) -> np.ndarray:
        return np.any(np.isfinite(lin.obs), axis=1)

    # ---------------------------------------------------------------- E-step
    def e_step(self, X: list[LineageTree] | None = None) -> list[TreePosterior]:
        X = self.X if X is None else X
        EL, offsets = get_scaled_Emission_Likelihoods(X, self.E)
        out = []
        for lin, el, off in zip(X, EL, offsets, strict=True):
            T = self.edge_matrices(lin)
            MSD = get_MSD(lin.tree, self.pi, T)
            logNF, beta = get_beta_and_logNF(lin.leaves_idx, lin.tree, T, MSD, el)
            gamma = get_gamma(lin.tree, T, MSD, beta)
            parents, children, xi = get_edge_posteriors(lin.tree, beta, MSD, gamma, T)
            LL = float(np.sum(logNF) + np.sum(off))
            out.append(TreePosterior(gamma, parents, children, xi, LL))
        return out

    # ---------------------------------------------------------------- M-step
    def m_step(self, post: list[TreePosterior]):
        K = self.K
        if self.fix_E is None:
            obs = np.vstack([x.obs for x in self.X])
            gam = np.vstack([p.gamma for p in post])
            for k in range(K):
                self.E[k].estimator(obs, gam[:, k])

        if self.mode == "ctmc":
            if self.fix_Q is None:
                t = np.concatenate([_lengths(x)[p.children] for x, p in zip(self.X, post, strict=True)])
                xi = np.concatenate([p.xi for p in post])
                self.Q = ctmc.M_step_Q(self.Q, t, xi)
        else:
            numer = np.full((K, K), 0.1 / K) + sum(p.xi.sum(axis=0) for p in post)
            self.T = numer / numer.sum(axis=1, keepdims=True)

        if self.pi_mode == "roots":
            roots = np.sum([p.gamma[0] for p in post], axis=0) + 1.0 / K
            self.pi = roots / roots.sum()
        elif self.mode == "ctmc":
            self.pi = ctmc.stationary(self.Q)
        else:
            w, v = np.linalg.eig(self.T.T)
            s = np.real(v[:, np.argmin(np.abs(w - 1))])
            self.pi = s / s.sum()

    # ---------------------------------------------------------------- fitting
    def init_emissions(self):
        """k-means on the observed leaves, then per-cluster mean and variance."""
        if self.fix_E is not None:
            return
        obs = np.vstack([x.obs for x in self.X])
        rows = np.all(np.isfinite(obs), axis=1)
        km = KMeans(self.K, n_init=1, random_state=int(self.rng.integers(2**31))).fit(obs[rows])
        for k in range(self.K):
            w = (km.labels_ == k).astype(float)
            self.E[k].estimator(obs[rows], w + 1e-6)

    def fit(self, tol: float = 1e-4, max_iter: int = 300, init: bool = True, verbose: bool = False):
        """EM until the per-observed-leaf log-likelihood improves by less than ``tol``."""
        if init:
            self.init_emissions()
        n_obs = max(1, sum(int(self.observed_leaf_mask(x).sum()) for x in self.X))
        post = self.e_step()
        old = sum(p.LL for p in post)
        self.LL_trace = [old]
        for it in range(max_iter):
            self.m_step(post)
            post = self.e_step()
            new = sum(p.LL for p in post)
            self.LL_trace.append(new)
            if verbose:
                print(it, new)
            if (new - old) / n_obs < tol:
                break
            old = new
        self.posteriors = post
        self.LL = self.LL_trace[-1]
        return self

    # ---------------------------------------------------------------- summaries
    def num_parameters(self) -> int:
        dof = 0 if self.pi_mode == "stationary" else self.K - 1
        if self.fix_Q is None:
            dof += self.K * (self.K - 1)
        if self.fix_E is None:
            dof += sum(e.dof() for e in self.E)
        return dof

    def BIC(self) -> float:
        n_obs = sum(int(self.observed_leaf_mask(x).sum()) for x in self.X)
        return -2 * self.LL + np.log(n_obs) * self.num_parameters()

    def switch_summary(self, post: list[TreePosterior] | None = None) -> list[dict]:
        """Per-tree expected state changes from the edge-wise joint posterior.

        ``edge_changes[k, l]`` is the expected number of edges whose parent is in k and child in l (k != l);
        this is the posterior analog of a parsimony count. In ctmc mode ``jumps[k, l]`` additionally counts
        every k -> l jump along the branches, including ones that revert within an edge.
        """
        post = self.posteriors if post is None else post
        out = []
        for lin, p in zip(self.X, post, strict=True):
            joint = p.xi.sum(axis=0)
            ec = joint.copy()
            stay = np.trace(ec)
            np.fill_diagonal(ec, 0.0)
            per_edge = 1.0 - np.einsum("ekk->e", p.xi)  # P(parent state != child state) for each edge
            d = {
                "n_edges": len(p.children),
                "edge_joint": joint,
                "edge_changes": ec,
                "expected_changes": float(ec.sum()),
                "plasticity": float(ec.sum() / max(1, len(p.children))),
                "heritability": float(stay / max(1, len(p.children))),
                "per_edge_change": per_edge,
            }
            if self.mode == "ctmc":
                t = _lengths(lin)[p.children]
                dwell, jumps = ctmc.expected_statistics(self.Q, t, p.xi)
                d["jumps"] = jumps
                d["dwell"] = dwell
            out.append(d)
        return out

    def heldout_loglik(self, X_masked: list[LineageTree], X_full: list[LineageTree], lineage: bool = True) -> float:
        """Predictive log-density of leaves observed in ``X_full`` but NaN in ``X_masked``, under this fit.

        The posterior at a masked leaf given all other data is exactly its predictive state distribution,
        so the score is :math:`\\sum_j \\log \\sum_k P(z_j = k | Y_{-j}) p(y_j | z_j = k)`.
        With ``lineage=False`` the tree is ignored at prediction time (population state frequencies
        replace the posterior), so the difference between the two scores is the information the
        lineage carries about a cell's state.
        """
        post = self.e_step(X_masked)
        if not lineage:
            # Same emissions, but every masked cell gets the population-wide state frequencies
            # of the observed leaves: what the embedding alone predicts without the tree.
            obs_gam = np.vstack([p.gamma[self.observed_leaf_mask(x)] for x, p in zip(X_masked, post, strict=True)])
            freq = obs_gam.mean(axis=0)
        total = 0.0
        for lin_m, lin_f, p in zip(X_masked, X_full, post, strict=True):
            hidden = self.observed_leaf_mask(lin_f) & ~self.observed_leaf_mask(lin_m)
            if not np.any(hidden):
                continue
            logEL = np.column_stack([e.logpdf(lin_f.obs[hidden]) for e in self.E])
            w = p.gamma[hidden] if lineage else np.broadcast_to(freq, logEL.shape)
            total += float(np.sum(logsumexp(logEL + np.log(np.maximum(w, 1e-300)), axis=1)))
        return total


def fit_best(X, num_states, n_init=5, rng=None, **kwargs) -> PhyloHMM:
    """Fit from several random starts and keep the highest log-likelihood."""
    rng = np.random.default_rng(rng)
    fit_kwargs = {k: kwargs.pop(k) for k in ("tol", "max_iter") if k in kwargs}
    best = None
    for _ in range(n_init):
        m = PhyloHMM(X, num_states, rng=rng, **kwargs).fit(**fit_kwargs)
        if best is None or m.LL > best.LL:
            best = m
    assert best is not None
    return best


def mask_leaves(X: list[LineageTree], frac: float, rng=None) -> list[LineageTree]:
    """Copy the trees with a random fraction of the observed leaves set to NaN."""
    rng = np.random.default_rng(rng)
    out = []
    for lin in X:
        obs = lin.obs.copy()
        observed = np.nonzero(np.any(np.isfinite(obs), axis=1))[0]
        hide = observed[rng.random(observed.size) < frac]
        obs[hide] = np.nan
        new = LineageTree(lin.tree, lin.E, obs=obs, branch_lengths=lin.branch_lengths)
        new.names = lin.names
        out.append(new)
    return out
