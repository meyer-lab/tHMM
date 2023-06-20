"""To plot population level of lapatinib from the lineage trees."""

from ..Lineage_collections import AllLapatinib
from ..LineageTree import cell_to_parent, get_leaves_idx
from .common import getSetup
import numpy as np
import random

def makeFigure():
    """
    Makes figure 1.
    """
    # Get list of axis objects
    ax, f = getSetup((10, 3), (1, 4))
    control = AllLapatinib[0]
    c1 = AllLapatinib[1]
    c2 = AllLapatinib[2]
    c3 = AllLapatinib[3]

    ts = np.arange(96, step=0.5)

    Gs = leaves_to_root(control[0], ts)
    for lineage in control[1:]:
        Gs += leaves_to_root(lineage, ts)
    ax[0].plot(Gs[:, 0], label="G1")
    ax[0].plot(Gs[:, 1], label="SG2")
    ax[0].legend()

    Gs1 = leaves_to_root(c1[0], ts)
    for lineage in c1[1:]:
        Gs1 += leaves_to_root(lineage, ts)
    ax[1].plot(Gs1)

    Gs2 = leaves_to_root(c2[0], ts)
    for lineage in c2[1:]:
        Gs2 += leaves_to_root(lineage, ts)
    ax[2].plot(Gs2)

    Gs3 = leaves_to_root(c3[0], ts)
    for lineage in c3[1:]:
        Gs3 += leaves_to_root(lineage, ts)
    ax[3].plot(Gs3)

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

            if t_cur > ts[-1]:
                return counts

            if bern_obs:
                return counts