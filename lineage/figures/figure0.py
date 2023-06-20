"""To plot population level of lapatinib from the lineage trees."""

from ..Lineage_collections import AllLapatinib, AllGemcitabine
from ..LineageTree import get_leaves_idx
from .common import getSetup
import numpy as np
import random

def makeFigure():
    """
    Makes figure 1.
    """
    # Get list of axis objects
    ax, f = getSetup((10, 6), (2, 4))

    lpt = ["Lapatinib, 25", "Lapatinib, 50", "Lapatinib 250"]
    gmc = ["Gemcitabine, 5", "Gemcitabine, 10", "Gemcitabine, 30"]
    plot_each_drug(f, ax[:4], AllLapatinib, lpt)
    plot_each_drug(f, ax[4:], AllGemcitabine, gmc)
    return f

def leaves_to_root(lin, ts):
    """A function to traverse from root to leaves 
    by randomly selecting a daughter cell at each generation forward, 
    and calculate the time-series counts at G1 and S/G2."""
    counts = np.zeros((ts.size, 2)) # G1 and S/G2

    t_cur = 0.0

    # randomly choose one of leaf cells.
    leaves_idx = get_leaves_idx(lin.output_lineage)
    lineage = lin.output_lineage
    random_leaf_idx = random.choice(leaves_idx)

    cells = [lineage[random_leaf_idx]]

    # traverse back to root from the randomly chosen leaf
    for c in cells:
        if c.parent is not None:
            cells.append(c.parent)

    cells.reverse() # order from root to leaf

    for ii, cell in enumerate(cells): # the number of generations
        for phase in range(2):
            bern_obs = cell.obs[phase]
            gamma_obs = np.nan_to_num(cell.obs[phase+2])
            idx = (ts > t_cur) & (ts < t_cur + gamma_obs)
            counts[idx, phase] += 2 ** (ii+1)
            t_cur += gamma_obs
    return counts

def plot_each_drug(f, ax, all_drug, name_list):
    control = all_drug[0]
    c1 = all_drug[1]
    c2 = all_drug[2]
    c3 = all_drug[3]

    ts = np.arange(96, step=0.5)

    Gs = leaves_to_root(control[0], ts)
    for lineage in control[1:]:
        Gs += leaves_to_root(lineage, ts)
    ax[0].plot(Gs[:, 0], label="G1")
    ax[0].plot(Gs[:, 1], label="SG2")
    ax[0].set_title("Untreated")
    ax[0].set_xlabel("Time [0.5hr]")
    ax[0].set_ylabel("Cell counts")
    ax[0].legend()

    Gs1 = leaves_to_root(c1[0], ts)
    for lineage in c1[1:]:
        Gs1 += leaves_to_root(lineage, ts)
    ax[1].plot(Gs1[:, 0], label="G1")
    ax[1].plot(Gs1[:, 1], label="SG2")
    ax[1].set_title(name_list[0] + " nM")
    ax[1].set_xlabel("Time [0.5hr]")
    ax[1].set_ylabel("Cell counts")
    ax[1].legend()

    Gs2 = leaves_to_root(c2[0], ts)
    for lineage in c2[1:]:
        Gs2 += leaves_to_root(lineage, ts)
    ax[2].plot(Gs2[:, 0], label="G1")
    ax[2].plot(Gs2[:, 1], label="SG2")
    ax[2].set_title(name_list[1] + " nM")
    ax[2].set_xlabel("Time [0.5hr]")
    ax[2].set_ylabel("Cell counts")
    ax[2].legend()

    Gs3 = leaves_to_root(c3[0], ts)
    for lineage in c3[1:]:
        Gs3 += leaves_to_root(lineage, ts)
    ax[3].plot(Gs3[:, 0], label="G1")
    ax[3].plot(Gs3[:, 1], label="SG2")
    ax[3].set_title(name_list[2] + " nM")
    ax[3].set_xlabel("Time [0.5hr]")
    ax[3].set_ylabel("Cell counts")
    ax[3].legend()

    f.tight_layout()
