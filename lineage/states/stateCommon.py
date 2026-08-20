"""Common utilities used between states regardless of distribution."""

import warnings
from ctypes import CFUNCTYPE, c_double
from typing import Literal

import numpy as np
import numpy.typing as npt
from numba import njit
from numba.extending import get_cython_function_address
from scipy.optimize import Bounds, LinearConstraint, minimize

arr_type = npt.NDArray[np.float64]


warnings.filterwarnings("ignore", message="Values in x were outside bounds")


def basic_censor(cells: list):
    """
    Censors a cell if the cell's parent is censored.
    """
    for cell in cells[1:]:
        if not cell.parent.observed:
            cell.observed = False


def apply_censoring(
    full_lineage: list,
    censor_condition: int,
    desired_experiment_time: float,
    assign_times_fn,
    fate_censor_fn,
    time_censor_fn,
) -> list:
    """Applies temporal and fate censorship across a lineage of cells."""
    assign_times_fn(full_lineage)

    if censor_condition == 0:
        return full_lineage

    for cell in full_lineage:
        if censor_condition in (1, 3):
            fate_censor_fn(cell)
        if censor_condition in (2, 3):
            time_censor_fn(cell, desired_experiment_time)

    basic_censor(full_lineage)
    return [c for c in full_lineage if c.observed]


def bern_estimator(bern_obs: np.ndarray, gammas: np.ndarray):
    """A weighted estimator for a Bernoulli distribution."""
    assert bern_obs.shape == gammas.shape
    assert bern_obs.dtype == float
    assert gammas.dtype == float

    # Add a pseudocount
    numerator = np.sum(gammas[bern_obs == 1.0]) + 1.0
    denominator = np.sum(gammas[np.isfinite(bern_obs)]) + 2.0
    return numerator / denominator


addr = get_cython_function_address("scipy.special.cython_special", "gammaincc")
gammaincc = CFUNCTYPE(c_double, c_double, c_double)(addr)

addr = get_cython_function_address("scipy.special.cython_special", "gammaln")
gammaln = CFUNCTYPE(c_double, c_double)(addr)

addr = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_0psi")
psi = CFUNCTYPE(c_double, c_double)(addr)


@njit
def gamma_LL(logX: arr_type, gamma_obs: arr_type, time_cen: arr_type, gammas: arr_type, param_idx):
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

    for jj, cen in enumerate(time_cen):
        if cen == 0:
            gamP = gammaincc(x[0], gobs[jj])
            gamP = np.maximum(gamP, 1e-35)  # Clip if the probability hits exactly 0
            outt -= gammas[jj] * np.log(gamP)

    assert np.isfinite(outt)
    return outt


@njit
def gamma_LL_grad(logX: arr_type, gamma_obs: arr_type, time_cen: arr_type, gammas: arr_type, param_idx) -> arr_type:
    """Analytical gradient of gamma_LL with respect to logX."""
    x = np.exp(logX)
    a = x[0]
    glnA = gammaln(a)
    psiA = psi(a)
    gobs = gamma_obs / x[param_idx]

    grad = np.zeros_like(logX)

    # Uncensored contribution for theta_0 (log shape):
    grad[0] = -a * np.dot(gammas * time_cen, np.log(gobs) - psiA)

    # Scale parameter contributions for theta_k (log scales):
    for i in range(len(gamma_obs)):
        k = param_idx[i]
        if time_cen[i] == 1:
            grad[k] += gammas[i] * (a - gobs[i])
        else:
            gamP = gammaincc(a, gobs[i])
            gamP = np.maximum(gamP, 1e-35)
            pdf_term = np.exp(a * np.log(gobs[i]) - gobs[i] - glnA)
            grad[k] -= gammas[i] * (pdf_term / gamP)

    # Censored contribution for theta_0:
    h = 1e-6
    for jj, cen in enumerate(time_cen):
        if cen == 0:
            gamP_plus = np.maximum(gammaincc(a * np.exp(h), gobs[jj]), 1e-35)
            gamP_minus = np.maximum(gammaincc(a * np.exp(-h), gobs[jj]), 1e-35)
            dlogQ_dtheta0 = (np.log(gamP_plus) - np.log(gamP_minus)) / (2 * h)
            grad[0] -= gammas[jj] * dlogQ_dtheta0

    return grad


def gamma_estimator(
    gamma_obs: arr_type,
    time_cen: arr_type,
    gammas: arr_type,
    param_idx,
    x0: arr_type,
    phase: Literal["all", "G1", "G2"],
) -> arr_type:
    """
    This is a weighted estimator for the parameters of the Gamma distribution,
    estimating shared shape and separate scale parameters across drug concentrations.
    """
    arrgs = (
        gamma_obs,
        time_cen,
        gammas,
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

    res = minimize(
        gamma_LL,
        jac=gamma_LL_grad,
        x0=np.log(x0),
        args=arrgs,
        bounds=bnd,
        method="SLSQP",
        constraints=linc,
    )

    assert res.success
    return np.exp(res.x)
