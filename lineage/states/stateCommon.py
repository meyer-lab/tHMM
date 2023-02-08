""" Common utilities used between states regardless of distribution. """

import warnings
import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.special import gammaincc, gammaln


warnings.filterwarnings("ignore", message="Values in x were outside bounds")


def basic_censor(cell):
    """
    Censors a cell, its daughters, its sister, and
    it's sister's daughters if the cell's parent is
    censored.
    """
    if not cell.isRootParent():
        if not cell.parent.observed:

            cell.observed = False
            if not cell.isLeafBecauseTerminal():
                cell.left.observed = False
                cell.right.observed = False

            cell.get_sister().observed = False
            if not cell.get_sister().isLeafBecauseTerminal():
                cell.get_sister().left.observed = False
                cell.get_sister().right.observed = False


def bern_estimator(bern_obs: np.ndarray, gammas: np.ndarray):
    """A weighted estimator for a Bernoulli distribution."""
    assert bern_obs.shape == gammas.shape
    assert bern_obs.dtype == float
    assert gammas.dtype == float

    # Add a pseudocount
    numerator = np.sum(gammas[bern_obs == 1.0]) + 1.0
    denominator = np.sum(gammas[np.isfinite(bern_obs)]) + 2.0
    return numerator / denominator


def gamma_LL(
    logX: np.ndarray,
    gamma_obs: list[np.ndarray],
    time_cen: list[np.ndarray],
    gammas: list[np.ndarray],
):
    """Log-likelihood for the optionally censored Gamma distribution.
    The logX is the log transform of the parameters, in case of atonce estimation, it is [shape, scale1, scale2, scale3, scale4]."""
    x = np.exp(logX)
    glnA = gammaln(x[0])
    outt = np.zeros(len(x) - 1)
    for i in range(len(x) - 1):
        gobs = gamma_obs[i] / x[i + 1]
        outt[i] -= np.dot(
            gammas[i] * time_cen[i],
            (x[0] - 1.0) * np.log(gobs) - gobs - glnA - logX[i + 1],
        )

        jidx = np.argwhere(time_cen[i] == 0.0)
        if jidx.size > 0:
            gamP = gammaincc(x[0], gobs[jidx])
            gamP = np.maximum(gamP, 1e-35)  # Clip if the probability hits exactly 0
            outt[i] -= np.dot(gammas[i][jidx].T, np.log(gamP))

    assert np.all(np.isfinite(outt))
    return outt



def gamma_LL_diff(logX: np.ndarray,
    gamma_obs: list[np.ndarray],
    time_cen: list[np.ndarray],
    gammas: list[np.ndarray],
):
    """Calculating the diff for log likelihood of each parameter with step size = 1e-6."""
    base = gamma_LL(logX, gamma_obs, time_cen, gammas)

    logXshape = np.copy(logX)
    logXshape[0] += 1e-6
    
    shape_dx = gamma_LL(logXshape, gamma_obs, time_cen, gammas)

    logXscale = np.copy(logX)
    logXscale[1:] += 1e-6

    scale_dx = gamma_LL(logXscale, gamma_obs, time_cen, gammas)

    deriv = np.zeros_like(logX)
    deriv[0] = (np.sum(shape_dx) - np.sum(base)) / 1e-6
    deriv[1:] = (scale_dx - base) / 1e-6

    return np.sum(base), deriv


def gamma_estimator(
    gamma_obs: list[np.ndarray],
    time_cen: list[np.ndarray],
    gammas: list[np.ndarray],
    x0: np.ndarray,
    phase: str = "all",
) -> np.ndarray:
    """
    This is a weighted, closed-form estimator for two parameters
    of the Gamma distribution for estimating shared shape and separate scale parameters of several drug concentrations at once.
    In the phase-specific case, we have 3 linear constraints: scale1 > scale2, scale2 > scale3, scale3 > scale 4.
    In the non-specific case, we have only 1 constraint: scale1 > scale2 ==> A = np.array([1, 3])
    """
    # Handle no observations
    if np.sum([np.sum(g) for g in gammas]) < 0.1:
        gammas = [np.ones(g.size) for g in gammas]

    # Check shapes
    for i in range(len(gamma_obs)):
        assert gamma_obs[i].shape == time_cen[i].shape
        assert gamma_obs[i].shape == gammas[i].shape

    arrgs = (gamma_obs, time_cen, gammas)
    linc = list()

    if phase != "all":  # for constrained optimization
        A = np.zeros((3, 5))  # constraint Jacobian
        np.fill_diagonal(A[:, 1:], -1.0)
        np.fill_diagonal(A[:, 2:], 1.0)

        linc.append(
            {
                "type": "ineq",
                "fun": lambda x: np.diff(x)[1:],
                "jac": lambda _: A,
            }
        )

        if np.any(np.dot(A, x0) < 0.0):
            x0 = np.array([x0[0], x0[1], x0[1], x0[1], x0[1]])

    bnd = Bounds(np.full_like(x0, -3.5), np.full_like(x0, 6.0), keep_feasible=True)

    with np.errstate(all="raise"):
        res = minimize(
            gamma_LL_diff,
            jac=True,
            x0=np.log(x0),
            args=arrgs,
            bounds=bnd,
            method="SLSQP",
            constraints=linc,
        )

    assert res.success or ("maximum number of function evaluations" in res.message)
    return np.exp(res.x)
