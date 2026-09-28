"""Figure 22: heritability of palbociclib escape (issue #1016).

Draws the results written by ``python -m lineage.palbociclib_analysis``; run that first.
"""

import json

import numpy as np
import scipy.stats as sp
from sklearn.metrics import roc_curve
from statsmodels.duration.survfunc import SurvfuncRight

from ..palbociclib_analysis import CONDITIONS, OUTPUT
from ..palbociclib_loader import load_lineages
from .common import getSetup, subplotLabel

STATE_NAMES = ("sensitive", "tolerant")
COND_COLORS = ("#4c72b0", "#c44e52")
FEATURE_LABELS = {
    "mother_sensor": "mother CDK2 rate",
    "neg_mother_lifetime": "mother lifetime",
    "logistic": "logistic (both)",
    "tHMM": "tHMM",
}


def makeFigure():
    with open(OUTPUT) as f:
        res = json.load(f)
    pops = [load_lineages(c) for c in CONDITIONS]
    obs = [np.vstack([lin.obs for lin in pop]) for pop in pops]
    E = np.array(res["emissions"])  # [mu, sigma, kappa, lam] per state

    ax, f = getSetup((12, 7), (2, 3))

    # (a) CDK2 activation rate with the fitted state mixture for each condition.
    grid = np.linspace(-0.3, 1.2, 300)
    for d, (cond, o) in enumerate(zip(CONDITIONS, obs, strict=True)):
        x = o[np.isfinite(o[:, 0]), 0]
        ax[0].hist(x, bins=np.linspace(-0.3, 1.2, 60), density=True, alpha=0.35, color=COND_COLORS[d], label=cond)
        pi = res["pi"][d]
        mix = sum(pi[k] * sp.norm.pdf(grid, E[k, 0], E[k, 1]) for k in range(len(pi)))
        ax[0].plot(grid, mix, color=COND_COLORS[d])
    ax[0].set_xlabel("max CDK2 activation rate in G1 (1/h)")
    ax[0].set_ylabel("density")
    ax[0].legend()

    # (b) Kaplan-Meier of lifetimes vs the fitted Weibull mixture.
    tgrid = np.linspace(0, 25, 200)
    for d, (cond, o) in enumerate(zip(CONDITIONS, obs, strict=True)):
        m = np.isfinite(o[:, 1])
        km = SurvfuncRight(o[m, 1], o[m, 2])
        ax[1].step(km.surv_times, km.surv_prob, where="post", color=COND_COLORS[d], label=f"{cond} (KM)")
        pi = res["pi"][d]
        S = sum(pi[k] * np.exp(-((tgrid / E[k, 3]) ** E[k, 2])) for k in range(len(pi)))
        ax[1].plot(tgrid, S, "--", color=COND_COLORS[d], label=f"{cond} (tHMM)")
    ax[1].set_xlabel("time since birth (h)")
    ax[1].set_ylabel("fraction undivided")
    ax[1].set_ylim(0, 1.02)
    ax[1].legend(fontsize=7)

    # (c) Persistence probabilities with bootstrap 95% CIs.
    T = np.array(res["point"]["T"])
    lo, hi = (np.array(a) for a in res["bootstrap"]["T"])
    width = 0.35
    for k in range(2):
        xpos = np.arange(len(CONDITIONS)) + (k - 0.5) * width
        val = T[:, k, k]
        err = np.vstack([val - lo[:, k, k], hi[:, k, k] - val])
        ax[2].bar(xpos, val, width, yerr=err, capsize=3, label=f"$T_{{{k + 1}{k + 1}}}$ ({STATE_NAMES[k]})")
    ax[2].axhline(0.5, color="k", lw=0.5, ls=":")
    ax[2].set_xticks(
        np.arange(len(CONDITIONS)), [f"{c}\n{int(dd)} nM" for c, dd in zip(CONDITIONS, res["doses_nM"], strict=True)]
    )
    ax[2].set_ylabel("P(daughter keeps mother's state)")
    ax[2].set_ylim(0, 1)
    ax[2].legend(fontsize=7)

    # (d) Heritability beyond state frequency: T_22 against the stationary tolerant
    # fraction (the value T_22 takes with no memory), and the memory eigenvalue.
    pi_tol = np.array([p[-1] for p in res["pi"]])
    lam = np.array(res["point"]["memory_eigenvalue"])
    lam_lo, lam_hi = (np.array(a) for a in res["bootstrap"]["memory_eigenvalue"])
    xpos = np.arange(len(CONDITIONS))
    ax[3].bar(xpos - width / 2, T[:, 1, 1], width, label="$T_{22}$", color="#dd8452")
    ax[3].bar(xpos + width / 2, pi_tol, width, label=r"$\pi_2$ (no-memory $T_{22}$)", color="#bbbbbb")
    ax[3].errorbar(xpos, lam, yerr=np.vstack([lam - lam_lo, lam_hi - lam]), fmt="ko", capsize=3, label=r"$\lambda_2$")
    ax[3].set_xticks(xpos, list(CONDITIONS))
    ax[3].set_ylim(0, 1)
    ax[3].set_ylabel("probability / eigenvalue")
    ax[3].legend(fontsize=7)
    diff = res["bootstrap"]["diff"]["memory_eigenvalue"]
    ax[3].set_title(
        f"heritability LRT p = {res['lrt']['heritability']['p']:.1e}\n"
        rf"$\Delta\lambda_2$ = {diff['point']:.2f} [{diff['ci'][0]:.2f}, {diff['ci'][1]:.2f}]",
        fontsize=8,
    )

    # (e) Out-of-fold ROC, within the palbociclib dose, for forecasting daughter escape
    # from the mother alone. (Pooled across doses, any score that tracks the dose would
    # look predictive, since escape is far more common in control.)
    y = np.array(res["roc_y"])
    dose = np.array(res["roc_dose"])
    d = len(CONDITIONS) - 1
    sel = dose == d
    for feat, label in FEATURE_LABELS.items():
        fpr, tpr, _ = roc_curve(y[sel], np.array(res["roc_scores"][feat])[sel])
        a = res["auc_by_dose"][d][feat]
        ctrl = res["auc_by_dose"][0][feat]["auc"]
        ax[4].plot(fpr, tpr, label=f"{label}: {a['auc']:.2f} [{a['lo']:.2f}, {a['hi']:.2f}] (ctrl {ctrl:.2f})")
    ax[4].plot([0, 1], [0, 1], "k:", lw=0.5)
    ax[4].set_xlabel("false positive rate")
    ax[4].set_ylabel("true positive rate")
    ax[4].set_title(
        f"{CONDITIONS[d]}: daughter escape within {res['horizon_h']:.0f} h (n = {res['pairs']['n_by_dose'][d]})",
        fontsize=8,
    )
    ax[4].legend(fontsize=6, loc="lower right")

    # (f) Number of states.
    sel = res["state_selection"]
    ax[5].plot([s["states"] for s in sel], [s["BIC"] for s in sel], "o-")
    ax[5].set_xlabel("number of states")
    ax[5].set_ylabel("BIC")
    ax[5].set_xticks([s["states"] for s in sel])

    subplotLabel(ax)
    return f
