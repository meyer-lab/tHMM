"""Multivariate Gaussian emissions for continuous per-cell embeddings (e.g. PCA or scVI latents)."""

import numpy as np


class StateDistribution:
    r"""
    Gaussian emission with diagonal-plus-ridge covariance, :math:`\Sigma_k = \mathrm{diag}(\sigma_k^2) + \epsilon I`.

    Rows of ``x`` that are entirely NaN are treated as unobserved (e.g. the internal nodes of a
    reconstructed phylogeny, or held-out leaves) and have log-likelihood 0, so their emission
    likelihood is 1. Individually missing dimensions are marginalized out.
    """

    def __init__(self, mean: np.ndarray | None = None, var: np.ndarray | None = None, ridge: float = 1e-3, dim=None):
        if mean is None:
            assert dim is not None, "Either a mean or the dimension must be given."
            mean = np.zeros(dim)
        mean = np.asarray(mean, dtype=float)
        var = np.ones_like(mean) if var is None else np.asarray(var, dtype=float)
        assert mean.shape == var.shape
        self.ridge = ridge
        self.params = np.concatenate([mean, var])

    @property
    def dim(self) -> int:
        return self.params.size // 2

    @property
    def mean(self) -> np.ndarray:
        return self.params[: self.dim]

    @property
    def var(self) -> np.ndarray:
        """The ridge is carried inside the stored variance, so this is the full diagonal of :math:`\\Sigma_k`."""
        return self.params[self.dim :]

    def rvs(self, size: int, rng=None) -> tuple[np.ndarray, ...]:
        """Draw ``size`` observations, returned one array per dimension like the other emission classes."""
        rng = np.random.default_rng(rng)
        draws = rng.normal(self.mean, np.sqrt(self.var), size=(size, self.dim))
        return tuple(draws.T)

    def dist(self, other) -> float:
        """Euclidean distance between the state means."""
        assert isinstance(self, type(other))
        return float(np.linalg.norm(self.mean - other.mean))

    def dof(self) -> int:
        """A mean and a variance per dimension."""
        return 2 * self.dim

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """Log-density of each row, marginalizing out NaN entries. All-NaN rows return 0."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        observed = np.isfinite(x)
        resid = np.where(observed, x - self.mean, 0.0)
        per_dim = -0.5 * (np.log(2 * np.pi * self.var) + resid**2 / self.var)
        return np.sum(np.where(observed, per_dim, 0.0), axis=1)

    def estimator(self, x: np.ndarray, gammas: np.ndarray):
        """Weighted maximum-likelihood mean and variance. Unobserved entries carry no weight."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        observed = np.isfinite(x)
        w = gammas[:, np.newaxis] * observed
        wsum = np.sum(w, axis=0)
        if np.any(wsum <= 1e-12):
            # A state with (almost) no responsibility keeps its old parameters on those dimensions.
            wsum = np.where(wsum <= 1e-12, np.nan, wsum)
        xz = np.where(observed, x, 0.0)
        mean = np.sum(w * xz, axis=0) / wsum
        var = np.sum(w * (xz - np.nan_to_num(mean)) ** 2, axis=0) / wsum + self.ridge
        mean = np.where(np.isfinite(mean), mean, self.mean)
        var = np.where(np.isfinite(var), var, self.var)
        self.params = np.concatenate([mean, var])
