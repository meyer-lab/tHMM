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

A duration can also be *truncated*: only seen because it fell in a window
:math:`(a_i, b_i)`, as for a cell that enters the data only by dividing inside the movie.
Its likelihood is then conditional on the window,

.. math::

    \\frac{f(t_i)^{\\delta_i}\\,[S(t_i) - S(b_i)]^{1 - \\delta_i}}{S(a_i) - S(b_i)},

which no longer lets the scale be profiled out in closed form. When any weighted duration
is truncated, :func:`weibull_estimator` maximizes the likelihood numerically over
:math:`(\\log\\kappa, \\log\\lambda)`, starting from the closed-form fit.
"""

import numpy as np
from scipy.optimize import minimize

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


def _cum_hazard(t: np.ndarray, kappa: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative hazard :math:`H = (t/\\lambda)^\\kappa` and its gradient in
    :math:`(\\log\\kappa, \\log\\lambda)`; ``t`` may be 0 or infinite."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        z = np.log(t) - np.log(lam)
        H = np.exp(kappa * z)
        dH = np.stack([np.where(np.isfinite(z) & (H > 0.0), kappa * z * H, 0.0), -kappa * H], axis=-1)
    return H, dH


def _log_sf_diff(Hu, dHu, Hv, dHv) -> tuple[np.ndarray, np.ndarray]:
    """:math:`\\log[S(u) - S(v)]` for :math:`u < v` (``v`` may be infinite), and its gradient."""
    d = Hu - Hv
    one_minus_r = -np.expm1(d)
    dHv = np.where(np.isfinite(Hv)[:, None], dHv, 0.0)
    val = -Hu + np.log(one_minus_r)
    grad = (-dHu + np.exp(d)[:, None] * dHv) / one_minus_r[:, None]
    return val, grad


def weibull_logterms(
    t: np.ndarray, events: np.ndarray, lo: np.ndarray, hi: np.ndarray, kappa: float, lam: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-duration Weibull log likelihood, right-censored and truncated to ``(lo, hi)``.

    ``lo = 0`` and ``hi = inf`` give the plain censored likelihood.

    :return: the log likelihoods, and their gradients in :math:`(\\log\\kappa, \\log\\lambda)`
    """
    t = np.clip(t, TIME_FLOOR, None)
    Ht, dHt = _cum_hazard(t, kappa, lam)
    z = np.log(t) - np.log(lam)

    val = np.empty(t.size)
    grad = np.empty((t.size, 2))
    ev = events == 1.0
    val[ev] = np.log(kappa) - np.log(lam) + (kappa - 1.0) * z[ev] - Ht[ev]
    grad[ev] = np.column_stack([1.0 + kappa * z[ev], np.full(np.sum(ev), -kappa)]) - dHt[ev]

    Hhi, dHhi = _cum_hazard(hi, kappa, lam)
    cen = ~ev
    val[cen], grad[cen] = _log_sf_diff(Ht[cen], dHt[cen], Hhi[cen], dHhi[cen])

    trunc = (lo > 0.0) | np.isfinite(hi)
    if np.any(trunc):
        Hlo, dHlo = _cum_hazard(lo[trunc], kappa, lam)
        den_val, den_grad = _log_sf_diff(Hlo, dHlo, Hhi[trunc], dHhi[trunc])
        val[trunc] -= den_val
        grad[trunc] -= den_grad
    return val, grad


def weibull_estimator(
    t: np.ndarray,
    events: np.ndarray,
    weights: np.ndarray,
    kappa0: float = 1.0,
    lo: np.ndarray | None = None,
    hi: np.ndarray | None = None,
    lam0: float | None = None,
) -> tuple[float, float]:
    """Weighted MLE of a right-censored, optionally truncated, Weibull distribution.

    :param t: durations, all nonnegative and finite
    :param events: 1 where the event was observed, 0 where the duration is right-censored
    :param weights: nonnegative observation weights (e.g. posterior state probabilities)
    :param kappa0: starting shape, typically the previous M step's value
    :param lo: lower truncation bounds (0 for none), with ``lo < t``
    :param hi: upper truncation bounds (inf for none), with ``t <= hi``
    :param lam0: previous scale. With truncation the fit is numerical, and starting from
        ``(kappa0, lam0)`` when that beats the closed-form start keeps EM monotone.
    :return: ``(kappa, lam)``, the shape and scale
    """
    t = np.asarray(t, dtype=float)
    events = np.asarray(events, dtype=float)
    w = np.asarray(weights, dtype=float)
    lo = np.zeros_like(t) if lo is None else np.asarray(lo, dtype=float)
    hi = np.full_like(t, np.inf) if hi is None else np.asarray(hi, dtype=float)
    assert t.shape == events.shape == w.shape == lo.shape == hi.shape
    assert np.all(np.isfinite(t)) and np.all(t >= 0.0)
    assert np.all(w >= 0.0)

    keep = w > 0.0
    t, events, w, lo, hi = t[keep], events[keep], w[keep], lo[keep], hi[keep]
    if t.size == 0:
        return float(kappa0), 1.0

    kappa, lam = _closed_form(np.clip(t, TIME_FLOOR, None), events, w, kappa0)
    if np.all(lo <= 0.0) and np.all(np.isinf(hi)):
        return kappa, lam
    return _truncated_fit(t, events, w, lo, hi, [(kappa, lam), (kappa0, lam0)])


def _truncated_fit(t, events, w, lo, hi, starts) -> tuple[float, float]:
    """Numerical MLE over (log kappa, log lam) from the best of ``starts``."""
    t_ref = float(np.max(t))
    bounds = [np.log(KAPPA_BOUNDS), np.log(t_ref) + np.log([1.0 / MAX_SCALE_RATIO, MAX_SCALE_RATIO])]
    wsum = np.sum(w)

    def nll(theta):
        with np.errstate(all="ignore"):
            val, grad = weibull_logterms(t, events, lo, hi, np.exp(theta[0]), np.exp(theta[1]))
            f, g = -np.dot(w, val) / wsum, -(w @ grad) / wsum
        # An extreme step can overflow the hazard; reject it so the line search backs off.
        if not (np.isfinite(f) and np.all(np.isfinite(g))):
            return np.inf, np.zeros(2)
        return f, g

    thetas = [
        np.clip(np.log([k, lam]), [b[0] for b in bounds], [b[1] for b in bounds])
        for k, lam in starts
        if lam is not None
    ]
    theta0 = min(thetas, key=lambda th: nll(th)[0])
    res = minimize(nll, theta0, jac=True, method="L-BFGS-B", bounds=bounds)
    theta = res.x if res.fun <= nll(theta0)[0] else theta0
    return float(np.exp(theta[0])), float(np.exp(theta[1]))


def _closed_form(t: np.ndarray, events: np.ndarray, w: np.ndarray, kappa0: float) -> tuple[float, float]:
    """Censored (untruncated) MLE, with the scale profiled out."""
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
