"""Forecasting a daughter's arrest or escape from its mother's phenotype.

At the moment a mother divides, its whole life is known: its biosensor reading
:math:`x_p` and its lifetime :math:`t_p`. The question is whether that is enough to
predict, before the daughter has done anything, whether the daughter will escape the
drug within a horizon ``H`` or stay arrested for at least that long. "Escape" is by
default a division within ``H``; a data set can instead supply its own escape time per
cell (e.g. when the CDK2 sensor first reaches S-phase levels) in an extra observation
column. A daughter whose tracking ends before ``H`` without escaping has no defined
outcome and is left out.

Four forecasters are compared by ROC AUC, each scored on lineages it was not fit to:

* the mother's biosensor reading alone;
* the mother's lifetime alone (shorter predicts escape, so it enters negated);
* a logistic regression on both;
* the tHMM: the mother's state posterior from her own observation, pushed through the
  fitted transition matrix, :math:`\\sum_k P(z_p = k \\mid x_p, t_p)\\, T_{k,c}`, the
  probability that the daughter is in the fastest-cycling state ``c``.
"""

import numpy as np
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .heritability import order_by_lifetime, refit

FEATURES = ("mother_sensor", "neg_mother_lifetime", "logistic", "tHMM")


def daughter_outcome(t: np.ndarray, delta: np.ndarray, horizon: float, escape: np.ndarray | None = None) -> np.ndarray:
    """1 if the daughter escaped within ``horizon``, 0 if it was followed past ``horizon``
    without escaping, NaN if it was lost before either could be decided.

    :param t: how long each daughter was followed
    :param delta: 1 where the daughter's division was seen
    :param escape: time of escape for each daughter, NaN if never seen; defaults to the
        division time
    """
    if escape is None:
        escape = np.where(delta == 1.0, t, np.nan)
    out = np.full(t.shape, np.nan)
    out[t > horizon] = 0.0
    out[escape <= horizon] = 1.0
    return out


def mother_daughter_pairs(
    pops_by_dose: list[list], horizon: float, escape_col: int | None = None
) -> dict[str, np.ndarray]:
    """Every mother-daughter pair with a decided daughter outcome and a fully observed
    mother, across all doses.

    :param escape_col: observation column holding each cell's escape time (see
        :func:`daughter_outcome`); by default escape is division

    :return: dict of equal-length arrays: ``dose`` index, ``lineage`` (a global id, used
        to keep a lineage's pairs in one cross-validation fold), ``mother`` and ``daughter``
        cell indices within the lineage, ``x_p``, ``t_p``, and the outcome ``y``
    """
    rows = []
    gid = 0
    for d, pop in enumerate(pops_by_dose):
        for li, lin in enumerate(pop):
            parents, daughters = lin.edges
            esc = None if escape_col is None else lin.obs[daughters, escape_col]
            y = daughter_outcome(lin.obs[daughters, 1], lin.obs[daughters, 2], horizon, esc)
            for p, c, yy in zip(parents, daughters, y, strict=True):
                if np.isfinite(yy) and np.all(np.isfinite(lin.obs[p, :2])):
                    rows.append((d, gid, li, p, c, lin.obs[p, 0], lin.obs[p, 1], yy))
            gid += 1

    cols = ("dose", "lineage", "lineage_idx", "mother", "daughter", "x_p", "t_p", "y")
    arr = np.array(rows, dtype=float).reshape(-1, len(cols))
    out = {c: arr[:, i] for i, c in enumerate(cols)}
    for c in ("dose", "lineage", "lineage_idx", "mother", "daughter"):
        out[c] = out[c].astype(int)
    return out


def thmm_forecast(tHMMobj, x_p: np.ndarray, t_p: np.ndarray) -> np.ndarray:
    """P(daughter is in the fastest-cycling state | mother's own observation) under a fitted model.

    The mother is known to have divided, so her lifetime enters as an observed event.
    With two states this ranks daughters identically to the model's probability of a
    division within any fixed horizon, so it serves for any escape definition.
    """
    E, T, pi = tHMMobj.estimate.E, tHMMobj.estimate.T, tHMMobj.estimate.pi
    obs = np.column_stack([x_p, t_p, np.ones_like(t_p)])
    log_joint = np.column_stack([np.log(pi[k]) + E[k].logpdf(obs) for k in range(len(E))])
    post = np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))
    fastest = int(np.argmin([e.mean_lifetime() for e in E]))
    return post @ T[:, fastest]


def cross_validated_scores(
    sweep,
    pops_by_dose: list[list],
    horizon: float = 48.0,
    n_folds: int = 5,
    escape_col: int | None = None,
    rng=None,
) -> dict:
    """Out-of-fold forecasts for every pair, folding by lineage.

    The tHMM is refit on each training fold, warm-started from the full-data fit in
    ``sweep`` (a :class:`~lineage.heritability.DoseSweep`), so no daughter's outcome is ever
    scored by a model that saw it.

    :return: ``pairs`` (from :func:`mother_daughter_pairs`) and one array of scores per
        entry of :data:`FEATURES`, aligned with the pairs
    """
    rng = np.random.default_rng(rng)
    pairs = mother_daughter_pairs(pops_by_dose, horizon, escape_col)
    n = pairs["y"].size
    scores = {f: np.full(n, np.nan) for f in FEATURES}
    scores["mother_sensor"] = pairs["x_p"].copy()
    scores["neg_mother_lifetime"] = -pairs["t_p"]

    # Lineage ids are global across doses; map back to (dose, index in that dose).
    lin_dose = np.concatenate([np.full(len(pop), d) for d, pop in enumerate(pops_by_dose)])
    lin_idx = np.concatenate([np.arange(len(pop)) for pop in pops_by_dose])

    X = np.column_stack([pairs["x_p"], np.log(pairs["t_p"])])
    folds = GroupKFold(n_splits=n_folds).split(X, pairs["y"], groups=pairs["lineage"])
    for train, test in folds:
        # Escape rates differ enormously between doses, so the regression is fit within
        # each dose; a pooled fit would mostly learn which dose a mother came from.
        for d in range(len(pops_by_dose)):
            tr, te = train[pairs["dose"][train] == d], test[pairs["dose"][test] == d]
            if te.size == 0:
                continue
            if np.unique(pairs["y"][tr]).size < 2:
                scores["logistic"][te] = np.mean(pairs["y"][tr]) if tr.size else 0.5
                continue
            clf = make_pipeline(StandardScaler(), LogisticRegression())
            clf.fit(X[tr], pairs["y"][tr])
            scores["logistic"][te] = clf.predict_proba(X[te])[:, 1]

        test_lin = np.unique(pairs["lineage"][test])
        train_mask = ~np.isin(np.arange(lin_dose.size), test_lin)
        train_pops = [
            [pops_by_dose[d][i] for d_, i in zip(lin_dose[train_mask], lin_idx[train_mask], strict=True) if d_ == d]
            for d in range(len(pops_by_dose))
        ]
        objs, _, _ = refit(train_pops, sweep.per_dose, shared_T=False, rng=rng)
        order_by_lifetime(objs)
        for d, tO in enumerate(objs):
            sel = test[pairs["dose"][test] == d]
            scores["tHMM"][sel] = thmm_forecast(tO, pairs["x_p"][sel], pairs["t_p"][sel])

    return {"pairs": pairs, **scores}


def auc_by_dose(cv: dict, n_boot: int = 500, rng=None) -> list[dict[str, dict[str, float]]]:
    """:func:`auc_table` within each dose.

    Prefer this to the pooled table when escape rates differ between doses: pooled, any
    score that differs by dose looks predictive even if it carries no information about
    individual cells.
    """
    rng = np.random.default_rng(rng)
    dose = cv["pairs"]["dose"]
    out = []
    for d in np.unique(dose):
        sel = dose == d
        sub = {"pairs": {k: v[sel] for k, v in cv["pairs"].items()}, **{f: cv[f][sel] for f in FEATURES}}
        out.append(auc_table(sub, n_boot=n_boot, rng=rng))
    return out


def auc_table(cv: dict, n_boot: int = 500, rng=None) -> dict[str, dict[str, float]]:
    """AUC of every forecaster with a 95% CI from resampling whole lineages."""
    rng = np.random.default_rng(rng)
    y, groups = cv["pairs"]["y"], cv["pairs"]["lineage"]
    uniq = np.unique(groups)
    members = [np.nonzero(groups == g)[0] for g in uniq]

    out = {}
    for f in FEATURES:
        boots = []
        for _ in range(n_boot):
            idx = np.concatenate([members[i] for i in rng.integers(len(uniq), size=len(uniq))])
            if np.unique(y[idx]).size == 2:
                boots.append(roc_auc_score(y[idx], cv[f][idx]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[f] = {"auc": float(roc_auc_score(y, cv[f])), "lo": float(lo), "hi": float(hi)}
    return out
