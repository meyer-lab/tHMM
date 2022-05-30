""" Common utilities used between states regardless of distribution. """

import warnings
import numpy as np
from jax import jit, value_and_grad
from jax.nn import softplus
import jax.numpy as jnp
from jax.scipy.special import gammaincc
from jax.scipy.stats import gamma
from jax.config import config
from scipy.optimize import minimize, LinearConstraint

config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

warnings.filterwarnings("ignore", message="UserWarning: delta_grad == 0.0")


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


def nLL_atonce(x: jnp.ndarray, gamma_obs: list, time_cen: list, gammas: list):
    """ uses the nLL_atonce and passes the vector of scales and the shared shape parameter. """
    xx = softplus(x)
    outt = 0.0
    for i in range(len(x) - 1):
        outt -= jnp.dot(gammas[i] * time_cen[i], gamma.logpdf(gamma_obs[i], a=xx[0], scale=xx[i + 1]))
        outt -= jnp.dot(gammas[i] * (1 - time_cen[i]), gammaincc(xx[0], gamma_obs[i] / xx[i + 1]))

    return outt


nLL_atonceJ = jit(value_and_grad(nLL_atonce))


def softplus_inv(x):
    x = np.clip(x, -np.inf, 1000.0)
    x = x.astype(np.float128)
    x = np.log(np.exp(x) - 1.0)
    return x.astype(float)


def gamma_estimator(gamma_obs: list[np.ndarray], time_cen: list[np.ndarray], gammas: list[np.ndarray], x0=None):
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

    if x0 is None:
        x0 = np.array([200.0, 0.2, 0.4, 0.6, 0.8])

    if len(gamma_obs) == 4:  # for constrained optimization
        A = np.zeros((3, 5))  # is a matrix that contains the constraints. the number of rows shows the number of linear constraints.
        np.fill_diagonal(A[:, 1:], -1.0)
        np.fill_diagonal(A[:, 2:], 1.0)
        linc = [LinearConstraint(A, lb=np.zeros(3), ub=np.full(3, np.inf))]
        if np.allclose(np.dot(A, x0), 0.0):
            x0 = np.array([200.0, 0.2, 0.4, 0.6, 0.8])
    else:
        linc = list()

    res = minimize(nLL_atonceJ, x0=softplus_inv(x0), jac=True, args=arrgs, method="trust-constr", constraints=linc)
    assert res.success or ("maximum number of function evaluations is exceeded" in res.message)

    return softplus(res.x)
