"""Weighted maximum-likelihood fitting of a right-censored Weibull distribution.

For durations :math:`t_i` with event indicators :math:`\\delta_i` (1 when the event was
seen, 0 when the observation was right-censored) and posterior weights :math:`w_i`, the
Weibull log likelihood is

.. math::

    \\ell(\\kappa, \\lambda) = \\sum_i w_i \\delta_i \\left[\\log\\kappa - \\kappa\\log\\lambda
        + (\\kappa - 1)\\log t_i\\right] - \\sum_i w_i (t_i / \\lambda)^\\kappa.

Setting :math:`\\partial\\ell / \\partial\\lambda = 0` gives the scale in closed form,
:math:`\\lambda^\\kappa = \\sum_i w_i t_i^\\kappa / D` with :math:`D = \\sum_i w_i \\delta_i`.
Substituting it back leaves a one-dimensional score equation for the shape,

.. math::

    g(\\kappa) = \\frac{1}{\\kappa} + \\frac{\\sum_i w_i\\delta_i\\log t_i}{D}
        - \\frac{\\sum_i w_i t_i^\\kappa \\log t_i}{\\sum_i w_i t_i^\\kappa} = 0,

whose derivative :math:`g'(\\kappa) = -1/\\kappa^2 - \\mathrm{Var}_\\kappa(\\log t)` is strictly
negative. The root is therefore unique, and Newton-Raphson safeguarded by a bisection
bracket converges to it from any start.
"""

import numpy as np

#: Shape parameters are kept inside this range. The lower end is far below any
#: biologically meaningful hazard; the upper end catches the degenerate case where every
#: observed event happens at the same time, for which the MLE runs off to infinity.
KAPPA_BOUNDS = (1e-2, 100.0)

#: When no event carries weight, the MLE of the scale is infinite (the event never
#: happens). It is capped at this multiple of the longest observed duration, far enough
#: out that the survival function is indistinguishable from one over the data.
MAX_SCALE_RATIO = 1e4

TIME_FLOOR = 1e-10


def _score(kappa: float, logu: np.ndarray, w: np.ndarray, mean_event_logu: float) -> tuple[float, float]:
    """The profile score g(kappa) and its derivative, evaluated stably in log space."""
    s = kappa * logu
    e = w * np.exp(s - np.max(s))
    tot = np.sum(e)
    m1 = np.dot(e, logu) / tot
    m2 = np.dot(e, logu * logu) / tot
    g = 1.0 / kappa + mean_event_logu - m1
    dg = -1.0 / kappa**2 - max(m2 - m1 * m1, 0.0)
    return g, dg


def weibull_shape_root(
    logu: np.ndarray,
    w: np.ndarray,
    events: np.ndarray,
    kappa0: float = 1.0,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> float:
    """Solve the profile score equation for the Weibull shape by safeguarded Newton-Raphson.

    Newton steps are taken in :math:`\\log\\kappa`, where the score is much closer to
    linear; any step that leaves the current bracket is replaced by bisection.

    :param logu: log durations (any fixed rescaling of time is fine)
    :param w: nonnegative weights
    :param events: event indicators, 1 for observed and 0 for censored
    :param kappa0: starting shape
    :return: the maximum-likelihood shape, clipped to :data:`KAPPA_BOUNDS`
    """
    D = np.dot(w, events)
    mean_event_logu = np.dot(w * events, logu) / D

    lo, hi = np.log(KAPPA_BOUNDS[0]), np.log(KAPPA_BOUNDS[1])
    # g is decreasing: if it is still positive at the upper bound (or negative at the
    # lower bound) the root lies outside the admissible range.
    if _score(KAPPA_BOUNDS[1], logu, w, mean_event_logu)[0] >= 0.0:
        return KAPPA_BOUNDS[1]
    if _score(KAPPA_BOUNDS[0], logu, w, mean_event_logu)[0] <= 0.0:
        return KAPPA_BOUNDS[0]

    theta = float(np.clip(np.log(kappa0), lo, hi))
    for _ in range(max_iter):
        kappa = np.exp(theta)
        g, dg = _score(kappa, logu, w, mean_event_logu)
        if g > 0.0:
            lo = theta
        else:
            hi = theta

        # d g / d theta = kappa * dg
        step = g / (kappa * dg)
        theta_new = theta - step
        if not (lo < theta_new < hi):
            theta_new = 0.5 * (lo + hi)

        if abs(theta_new - theta) < tol:
            theta = theta_new
            break
        theta = theta_new

    return float(np.exp(theta))


def weibull_estimator(
    t: np.ndarray,
    events: np.ndarray,
    weights: np.ndarray,
    kappa0: float = 1.0,
) -> tuple[float, float]:
    """Weighted MLE of a right-censored Weibull distribution.

    :param t: durations, all nonnegative and finite
    :param events: 1 where the event was observed, 0 where the duration is right-censored
    :param weights: nonnegative observation weights (e.g. posterior state probabilities)
    :param kappa0: starting shape, typically the previous M step's value
    :return: ``(kappa, lam)``, the shape and scale
    """
    t = np.asarray(t, dtype=float)
    events = np.asarray(events, dtype=float)
    w = np.asarray(weights, dtype=float)
    assert t.shape == events.shape == w.shape
    assert np.all(np.isfinite(t)) and np.all(t >= 0.0)
    assert np.all(w >= 0.0)

    keep = w > 0.0
    t, events, w = np.clip(t[keep], TIME_FLOOR, None), events[keep], w[keep]
    if t.size == 0:
        return float(kappa0), 1.0

    # Work with durations relative to the longest one so that t**kappa cannot overflow.
    t_ref = float(np.max(t))
    logu = np.log(t / t_ref)

    D = float(np.dot(w, events))
    if D <= 1e-12 * np.sum(w):
        # No events at all: the cells in this state never do it within the window.
        return float(np.clip(kappa0, *KAPPA_BOUNDS)), MAX_SCALE_RATIO * t_ref

    kappa = weibull_shape_root(logu, w, events, kappa0=kappa0)

    s = kappa * logu
    smax = np.max(s)
    log_lam_u = (smax + np.log(np.dot(w, np.exp(s - smax))) - np.log(D)) / kappa
    lam = t_ref * min(np.exp(log_lam_u), MAX_SCALE_RATIO)
    return kappa, float(lam)


def weibull_loglik(t: np.ndarray, events: np.ndarray, weights: np.ndarray, kappa: float, lam: float) -> float:
    """Weighted right-censored Weibull log likelihood; used to check the estimator."""
    t = np.clip(np.asarray(t, dtype=float), TIME_FLOOR, None)
    z = np.log(t) - np.log(lam)
    logsf = -np.exp(kappa * z)
    logpdf = np.log(kappa) - np.log(lam) + (kappa - 1.0) * z + logsf
    return float(np.dot(weights, np.where(events == 1, logpdf, logsf)))


def gaussian_estimator(x: np.ndarray, weights: np.ndarray, min_sigma: float) -> tuple[float, float]:
    """Weighted MLE of a Gaussian's mean and standard deviation, with a floor on the latter."""
    wsum = np.sum(weights)
    mu = float(np.dot(weights, x) / wsum)
    var = float(np.dot(weights, (x - mu) ** 2) / wsum)
    return mu, max(np.sqrt(var), min_sigma)
