""" Common utilities used between states regardless of distribution. """

import warnings
import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize, Bounds, LinearConstraint
from scipy.special import gammaincc, gammaln

arr_type = npt.NDArray[np.float64]


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
    logX: arr_type, gamma_obs: arr_type, time_cen: arr_type, gammas: arr_type, param_idx
):
    """Log-likelihood for the optionally censored Gamma distribution.
    The logX is the log transform of the parameters, in case of atonce estimation, it is [shape, scale1, scale2, scale3, scale4].
    """
    x = np.exp(logX)
    glnA = gammaln(x[0])

    gobs = gamma_obs / x[param_idx]
    outt = -1.0 * np.dot(
        gammas * time_cen,
        (x[0] - 1.0) * np.log(gobs) - gobs - glnA - logX[param_idx],
    )

    jidx = time_cen == 0.0
    gamP = gammaincc(x[0], gobs[jidx])
    gamP = np.maximum(gamP, 1e-35)  # Clip if the probability hits exactly 0
    outt -= np.sum(gammas[jidx] * np.log(gamP))

    assert np.isfinite(outt)
    return outt


def gamma_estimator(
    gamma_obs: list[arr_type],
    time_cen: list[arr_type],
    gammas: list[arr_type],
    x0: arr_type,
    phase: str = "all",
) -> arr_type:
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

    param_idx = np.concatenate(
        [np.full(gam.size, ii + 1) for ii, gam in enumerate(gamma_obs)]
    )

    arrgs = (
        np.concatenate(gamma_obs),
        np.concatenate(time_cen),
        np.concatenate(gammas),
        param_idx,
    )

    if phase != "all":  # for constrained optimization
        A = np.zeros((3, 5))  # constraint Jacobian
        np.fill_diagonal(A[:, 1:], -1.0)
        np.fill_diagonal(A[:, 2:], 1.0)

        linc = LinearConstraint(A, lb=0.0, keep_feasible=False)
    else:
        linc = ()

    bnd = Bounds(-4.0, 7.0, keep_feasible=False)

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
