"""Death timing by phase, which is what sets the shape of each death clock.

The competing-risks emission in :mod:`lineage.states.StateDistributionCR` gives G1 a
one-parameter death clock with a constant hazard and G2 a two-parameter one. This
figure is the evidence for that split: pooled across the lapatinib and gemcitabine
conditions, G1 death times are essentially memoryless while G2 death times have a
strongly increasing hazard and arrive later than divisions do.
"""

import numpy as np
import scipy.stats as sp

from ..Lineage_collections import AllGemcitabine, AllLapatinib
from .common import getSetup


def gather() -> dict[str, np.ndarray]:
    """Pool per-phase event times across the lapatinib and gemcitabine conditions.

    Observation columns are ``[G1 fate, G2 fate, G1 time, G2 time, G1 cens, G2 cens]``,
    with a fate of 0 for death in that phase and 1 for surviving it. The shared control
    appears in both drug lists, so populations are de-duplicated by identity.
    """
    seen: dict[int, np.ndarray] = {}
    for drug in (AllLapatinib, AllGemcitabine):
        for population in drug:
            for lineage in population:
                seen.setdefault(id(lineage), lineage.obs)
    x = np.vstack(list(seen.values()))

    out = {}
    for name, fate, time, cens in (("G1", 0, 2, 4), ("G2", 1, 3, 5)):
        t = x[:, time]
        timed = np.isfinite(t) & (t > 0.0)
        out[f"{name} death"] = t[timed & (x[:, fate] == 0.0)]
        out[f"{name} division"] = t[timed & (x[:, fate] == 1.0) & (x[:, cens] == 1.0)]
    return out


def cumulative_hazard(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nelson-Aalen style cumulative hazard for a fully observed sample.

    On log-log axes a constant hazard plots as a line of slope 1; the slope is the
    Weibull shape, so anything steeper is a wear-out process.
    """
    ts = np.sort(t)
    n = len(ts)
    at_risk = n - np.arange(n)
    return ts, np.cumsum(1.0 / at_risk)


def makeFigure():
    """Compare the death and division timing of each cell-cycle phase."""
    data = gather()
    ax, f = getSetup((9, 3), (1, 3))

    # (a, b) survival of the death times against the best-fit exponential.
    for i, phase in enumerate(("G1", "G2")):
        t = data[f"{phase} death"]
        ts, _ = cumulative_hazard(t)
        ax[i].step(ts, 1.0 - np.arange(len(ts)) / len(ts), where="post", label="observed")
        ax[i].plot(ts, sp.expon(scale=t.mean()).sf(ts), "--", label="exponential")
        shape, _, scale = sp.weibull_min.fit(t, floc=0)
        ax[i].plot(ts, sp.weibull_min(shape, scale=scale).sf(ts), ":", label="Weibull")
        ax[i].set(
            title=f"{phase} death times (n={len(t)})",
            xlabel="time in phase [hr]",
            ylabel="fraction not yet dead",
        )
        ax[i].text(
            0.55,
            0.75,
            f"Weibull shape\n{shape:.2f}",
            transform=ax[i].transAxes,
            fontsize=9,
        )
        ax[i].legend(fontsize=7)

    # (c) cumulative hazards on log-log axes; slope is the Weibull shape.
    for label, style in (
        ("G1 death", "-"),
        ("G2 death", "-"),
        ("G1 division", "--"),
        ("G2 division", "--"),
    ):
        ts, H = cumulative_hazard(data[label])
        ax[2].loglog(ts, H, style, label=f"{label} (n={len(ts)})", linewidth=1)
    ref = np.array([1.0, 100.0])
    ax[2].loglog(ref, ref / 60.0, color="k", linewidth=0.5, label="slope 1 (constant hazard)")
    ax[2].set(title="Cumulative hazard", xlabel="time in phase [hr]", ylabel="cumulative hazard")
    ax[2].legend(fontsize=6)

    return f
