"""
File: figure1.py
Purpose: Generates figure 1.
Figure 1 is the tHMM model and its purpose.
"""

import random
from string import ascii_lowercase

from lineage.Lineage_collections import Gem10uM
from lineage.Lineage_collections import Gemcitabine_Control as control

from ..plotTree import plotLineage
from .common import getSetup


def makeFigure():
    """
    Makes figure 1.
    """
    # Get list of axis objects
    ax, f = getSetup((7.6, 3.1), (8, 3))
    figureMaker(ax)
    ax[0].text(
        -0.2,
        1.7,
        ascii_lowercase[0],
        transform=ax[0].transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
    )
    ax[1].text(
        -0.2,
        1.7,
        ascii_lowercase[1],
        transform=ax[1].transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
    )
    ax[2].text(
        -0.2,
        1.7,
        ascii_lowercase[2],
        transform=ax[2].transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
    )

    return f


def figureMaker(ax):
    """
    Makes figure 1.
    """
    indxs_gem = [random.randint(0, (len(Gem10uM) - 1)) for _ in range(7)]
    indxs_c1 = [0, 2, 6, 8, 9, 10, 24]
    indxs_c2 = [1, 4, 5, 13, 17, 18, 19]
    # titles
    ax[0].set_title("Control 1", fontsize=10)
    ax[1].set_title("Control 2", fontsize=10)
    ax[2].set_title("Gemcitabine 10 nM - random", fontsize=10)
    # lineages
    for i in range(7):
        plotLineage(control[indxs_c1[i]], ax[3 * i], censor=False, color=False)
        plotLineage(control[indxs_c2[i]], ax[3 * i + 1], censor=False, color=False)
        plotLineage(Gem10uM[indxs_gem[i]], ax[3 * i + 2], censor=False, color=False)

    for i in range(24):
        ax[i].axis("off")
