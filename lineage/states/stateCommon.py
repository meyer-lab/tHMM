""" Common utilities used between states regardless of distribution. """

import warnings
import numpy as np
import scipy.special as sc
from jax import jit, value_and_grad
import jax.numpy as jnp
from jax.scipy.special import gammaincc
from jax.scipy.stats import gamma
from jax.config import config
from scipy.optimize import toms748, minimize, Bounds, LinearConstraint, BFGS

config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
warnings.filterwarnings('ignore', r'delta_grad == 0.0.')


def nLL_sep(x, uncens_obs, uncens_gammas, cens_obs, cens_gammas):
    a, scale = x
    uncens = jnp.dot(uncens_gammas, gamma.logpdf(uncens_obs, a=a, scale=scale))
    cens = jnp.dot(cens_gammas, gammaincc(a, cens_obs / scale))
    return -1 * (uncens + cens)


GnLL_sep = jit(value_and_grad(nLL_sep))


def gamma_uncensored(gamma_obs, gammas):
    """ An uncensored gamma estimator. """
    # Handle no observations
    if np.sum(gammas) == 0.0:
        gammas = np.ones_like(gammas)

    gammaCor = np.average(gamma_obs, weights=gammas)
    s = np.log(gammaCor) - np.average(np.log(gamma_obs), weights=gammas)

    def f(k):
        return np.log(k) - sc.polygamma(0, k) - s

    flow = f(1.0)
    fhigh = f(100.0)
    if flow * fhigh > 0.0:
        if np.absolute(flow) < np.absolute(fhigh):
            a_hat0 = 1.0
        elif np.absolute(flow) > np.absolute(fhigh):
            a_hat0 = 100.0
        else:
            a_hat0 = 10.0
    else:
        a_hat0 = toms748(f, 1.0, 100.0)

    return [a_hat0, gammaCor / a_hat0]


def gamma_estimator(gamma_obs, time_cen, gammas, x0):
    """
    This is a weighted, closed-form estimator for two parameters
    of the Gamma distribution.
    """
    # Handle no observations
    if np.sum(gammas) == 0.0:
        gammas = np.ones_like(gammas)

    # If nothing is censored
    if np.all(time_cen == 1):
        return gamma_uncensored(gamma_obs, gammas)

    assert gammas.shape[0] == gamma_obs.shape[0]
    arrgs = (gamma_obs[time_cen == 1], gammas[time_cen == 1], gamma_obs[time_cen == 0], gammas[time_cen == 0])
    opt = {'gtol': 1e-12, 'ftol': 1e-12}
    bnd = (1.0, 100.0)

    # def GnLL(x, *args):
    #     val, grad = GnLL_sep(x, *args)
    #     return val, np.array(grad)

    res = minimize(GnLL_sep, x0, jac=True, bounds=(bnd, bnd), args=arrgs, method="trust-constr")
    if not res.success:
        print(res.message)
    assert (res.success is True) or ("maximum number of function evaluations is exceeded" in res.message)

    return res.x


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


def nLL_atonce(x, uncens_obs, uncens_gammas, cens_obs, cens_gammas):
    """ uses the nLL_atonce and passes the vector of scales and the shared shape parameter. """
    outt = 0.0
    for i in range(len(x)-1):
        outt += nLL_sep([x[0], x[1 + i]], uncens_obs[i], uncens_gammas[i], cens_obs[i], cens_gammas[i])

    return outt


nLL_atonceJ = jit(value_and_grad(nLL_atonce))


def gamma_estimator_atonce(gamma_obs, time_cen, gamas, x0=None, phase=True):
    """
    This is a weighted, closed-form estimator for two parameters
    of the Gamma distribution for estimating shared shape and separate scale parameters of several drug concentrations at once.
    In the phase-specific case, we have 3 linear constraints: scale1 > scale2, scale2 > scale3, scale3 > scale 4.
    In the non-specific case, we have only 1 constraint: scale1 > scale2 ==> A = np.array([1, 3])
    """
    # Handle no observations
    gammas = [np.ones_like(g) if np.sum(g) == 0.0 else g for g in gamas]

    for i, gamma in enumerate(gammas):
        assert gamma.shape[0] == gamma_obs[i].shape[0]
    arg1 = [np.squeeze(gamma_obs[i][time_cen[i] == 1]) for i in range(len(gamma_obs))]
    arg2 = [np.squeeze(gammas[i][time_cen[i] == 1]) for i in range(len(gamma_obs))]
    arg3 = [np.squeeze(gamma_obs[i][time_cen[i] == 0]) for i in range(len(gamma_obs))]
    arg4 = [np.squeeze(gammas[i][time_cen[i] == 0]) for i in range(len(gamma_obs))]

    arrgs = (arg1, arg2, arg3, arg4)

    if x0 is None:
        if phase:
            x0 = np.array([20.0, 2.0, 3.0, 4.0, 5.0])
        else:
            x0 = np.array([20.0, 2.0, 3.0])

    if phase: # phase-specific observations
        A = np.zeros((3, 5)) # is a matrix that contains the constraints. the number of rows shows the number of linear constraints.
        np.fill_diagonal(A[:, 1:], -1.0)
        np.fill_diagonal(A[:, 2:], 1.0)
        linc = LinearConstraint(A, lb=np.zeros(3), ub=np.ones(3) * 100.0)
        if np.allclose(np.dot(A, x0), 0.0):
            x0 = np.array([20.0, 1.0, 2.0, 3.0, 4.0])

    bnds = Bounds(lb=np.ones_like(x0) * 0.01, ub=np.ones_like(x0) * 100.0, keep_feasible=True)
    HH = BFGS()

    if phase:
        res = minimize(nLL_atonceJ, x0=x0, jac=True, hess=HH, args=arrgs, method="trust-constr", bounds=bnds, constraints=[linc])
        assert (res.success is True) or ("maximum number of function evaluations is exceeded" in res.message)
    else:
        res = minimize(nLL_atonceJ, x0=x0, jac=True, hess=HH, args=arrgs, bounds=bnds, method="trust-constr")
        assert (res.success is True) or ("maximum number of function evaluations is exceeded" in res.message)

    return res.x
