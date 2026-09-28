"""Figure 23: heritability of the CDK2 state under timed MEK or ERK inhibition.

Draws the results written by ``python -m lineage.mitogen_analysis``; run that first.
"""

import json

import numpy as np
import scipy.stats as sp

from ..mitogen_analysis import OUTPUT
from ..palbociclib_loader import ROOT_MODES
from .common import getSetup, subplotLabel

DRUG_COLORS = {"MEKi": "#4c72b0", "ERKi": "#c44e52"}
MODE_MARKERS = {"truncate": "o", "keep": "s", "drop": "^"}
RELATIVES = {"mother_daughter": "mother-daughter", "sisters": "sisters", "cousins": "cousins"}


def pulse_labels(pulses) -> list[str]:
    return ["none" if p == 0 else "on" if np.isinf(p) else f"{p:g} h" for p in pulses]


def makeFigure():
    with open(OUTPUT) as f:
        res = json.load(f)
    drugs, pulses = res["drugs"], np.array(res["pulses_h"], dtype=float)
    labels = pulse_labels(pulses)
    xpos = np.arange(len(pulses))
    offset = {d: (i - 0.5) * 0.2 for i, d in enumerate(drugs)}

    ax, f = getSetup((12, 7), (2, 3))

    # (a) What the two states are: biosensor densities, labelled with mean lifetimes.
    grid = np.linspace(-0.3, 1.2, 300)
    for d in drugs:
        for k, (e, life) in enumerate(zip(res[d]["emissions"], res[d]["mean_lifetime_h"], strict=True)):
            ax[0].plot(
                grid,
                sp.norm.pdf(grid, e[0], e[1]),
                color=DRUG_COLORS[d],
                ls="-" if k else "--",
                label=f"{d} state {k + 1}: mean lifetime {life:.0f} h",
            )
    ax[0].set_xlabel("max CDK2 activation rate in G1 (1/h)")
    ax[0].set_ylabel("density")
    ax[0].legend(fontsize=6)

    # (b) Division by 20 h among cells born into each condition (not selected on their
    # outcome): observed against the fit, for each treatment of the roots' lifetimes.
    for d in drugs:
        fits = {"truncate": res[d]["divided_by_20h"]} | {
            m: res[d]["root_modes"][m]["divided_by_20h"] for m in ROOT_MODES[1:]
        }
        for mode in ROOT_MODES:
            obs = [c["observed"] for c in fits[mode]]
            mod = [c["model"] for c in fits[mode]]
            ax[1].scatter(obs, mod, marker=MODE_MARKERS[mode], color=DRUG_COLORS[d], s=18, label=f"{d}, roots {mode}")
    ax[1].plot([0, 1], [0, 1], "k:", lw=0.5)
    ax[1].set_xlabel("observed fraction divided by 20 h (KM)")
    ax[1].set_ylabel("predicted")
    ax[1].legend(fontsize=6)

    # (c) Tolerant-state persistence against exposure, with bootstrap 95% CIs.
    # (d) The memory eigenvalue, which is zero when daughters' states do not depend on
    # their mother's.
    for d in drugs:
        for a, key in ((ax[2], "T_tolerant"), (ax[3], "memory_eigenvalue")):
            val = np.array(res[d]["point"][key])
            lo, hi = (np.array(b) for b in res[d]["bootstrap"][key])
            a.errorbar(
                xpos + offset[d],
                val,
                yerr=np.vstack([val - lo, hi - val]),
                fmt="o-",
                capsize=3,
                color=DRUG_COLORS[d],
                label=d,
            )
    ax[2].set_ylabel("$T_{22}$: P(tolerant daughter | tolerant mother)")
    ax[2].set_ylim(0, 1)
    ax[3].axhline(0.0, color="k", lw=0.5)
    ax[3].set_ylabel(r"memory eigenvalue $\lambda_2$")
    ax[3].set_ylim(-1, 1)
    ax[3].set_title(
        "heritability LRT: " + ", ".join(f"{d} p = {res[d]['lrt']['heritability']['p']:.1e}" for d in drugs), fontsize=8
    )
    for a in ax[2:4]:
        a.set_xticks(xpos, labels)
        a.set_xlabel("exposure")
        a.legend(fontsize=7)

    # (e) Model-free check: CDK2-rate correlation between relatives born into the
    # condition, for no drug and for each drug left on.
    groups = [("none", res[drugs[0]]["relatives"][0])] + [(f"{d} on", res[d]["relatives"][-1]) for d in drugs]
    width = 0.8 / len(groups)
    for i, (name, rel) in enumerate(groups):
        r = [rel[k]["r"] for k in RELATIVES]
        ax[4].bar(np.arange(len(RELATIVES)) + (i - (len(groups) - 1) / 2) * width, r, width, label=name)
    ax[4].axhline(0.0, color="k", lw=0.5)
    ax[4].set_xticks(np.arange(len(RELATIVES)), list(RELATIVES.values()))
    ax[4].set_ylabel("Pearson r of CDK2 activation rate")
    ax[4].set_ylim(-0.2, 1.0)
    ax[4].legend(fontsize=7)

    # (f) Generations below the root, per condition: the depth this data adds over the
    # palbociclib movies.
    for d in drugs:
        depth = res[d]["depth_counts"]
        frac2 = [sum(c[2:]) / sum(c[1:]) for c in depth]
        frac3 = [sum(c[3:]) / sum(c[1:]) for c in depth]
        ax[5].plot(xpos + offset[d], frac2, "o-", color=DRUG_COLORS[d], label=f"{d}: 2+ generations")
        ax[5].plot(xpos + offset[d], frac3, "s--", color=DRUG_COLORS[d], label=f"{d}: 3+ generations")
    ax[5].set_xticks(xpos, labels)
    ax[5].set_xlabel("exposure")
    ax[5].set_ylabel("fraction of cells born into the condition")
    ax[5].set_ylim(0, None)
    ax[5].legend(fontsize=6)

    subplotLabel(ax)
    return f
