""" Common utilities used between states regardless of distribution. """

import warnings
import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize, Bounds
from scipy.special import gammaincc, gammaln


warnings.filterwarnings("ignore", message="Values in x were outside bounds")


def basic_censor(cells: list):
    """
    Censors a cell if the cell's parent is censored.
    """
    for cell in cells:
        if not cell.isRootParent():
            if not cell.parent.observed:
                cell.observed = False


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
    logX: npt.NDArray[np.float64],
    gamma_obs: list[npt.NDArray[np.float64]],
    time_cen: list[npt.NDArray[np.float64]],
    gammas: list[npt.NDArray[np.float64]],
):
    """Log-likelihood for the optionally censored Gamma distribution.
    The logX is the log transform of the parameters, in case of atonce estimation, it is [shape, scale1, scale2, scale3, scale4].
    """
    x = np.exp(logX)
    glnA = gammaln(x[0])
    outt = 0.0
    for i in range(len(x) - 1):
        gobs = gamma_obs[i] / x[i + 1]
        outt -= np.dot(
            gammas[i] * time_cen[i],
            (x[0] - 1.0) * np.log(gobs) - gobs - glnA - logX[i + 1],
        )

        jidx = np.argwhere(time_cen[i] == 0.0)
        for idxx in jidx[:, 0]:
            gamP = gammaincc(x[0], gobs[idxx])
            gamP = np.maximum(gamP, 1e-35)  # Clip if the probability hits exactly 0
            outt -= gammas[i][idxx] * np.log(gamP)

    assert np.isfinite(outt)
    return outt


def gamma_estimator(
    gamma_obs: list[np.ndarray],
    time_cen: list[np.ndarray],
    gammas: list[np.ndarray],
    x0: np.ndarray,
    phase: str = "all",
) -> npt.NDArray[np.float64]:
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

    arrgs = (list(gamma_obs), list(time_cen), list(gammas))
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

    bnd = Bounds(-4.0, 7.0, keep_feasible=True)

    with np.errstate(all="raise"):
        res = minimize(
            gamma_LL,
            jac="2-point",
            x0=np.log(x0),
            args=arrgs,
            bounds=bnd,
            method="SLSQP",
            constraints=linc,
        )

    assert res.success
    return np.exp(res.x)
