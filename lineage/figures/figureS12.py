""" This is a file to put together 4 conditions of gemcitabine together. """

from string import ascii_lowercase
from .common import getSetup, sort_lins
from ..plotTree import plotLineage
from ..Analyze import Analyze_list
from ..Lineage_collections import AllGemcitabine

num_states = 5
gemc_tHMMobj_list = Analyze_list(AllGemcitabine, num_states, write_states=True)[0]

for i in range(4):
    gemc_tHMMobj_list[i].X = sort_lins(gemc_tHMMobj_list[i])


def makeFigure():
    """
    Makes figure 150.
    """

    titles = ["Control", "Gemcitabine 5 nM", "Gemcitabine 10 nM", "Gemcitabine 30 nM"]
    ax, f = getSetup((15, 35), (150, 4))

    for i in range(150):
        ax[4 * i].axis("off")
        ax[4 * i + 1].axis("off")
        ax[4 * i + 2].axis("off")
        ax[4 * i + 3].axis("off")
        plotLineage(gemc_tHMMobj_list[0].X[i], ax[4 * i])
        plotLineage(gemc_tHMMobj_list[1].X[i], ax[4 * i + 1])
        plotLineage(gemc_tHMMobj_list[2].X[i], ax[4 * i + 2])
        plotLineage(gemc_tHMMobj_list[3].X[i], ax[4 * i + 3])

    for i in range(4):
        ax[i].axis("off")
        ax[i].text(
            -0.2,
            1.55,
            ascii_lowercase[i],
            transform=ax[i].transAxes,
            fontsize=20,
            fontweight="bold",
            va="top",
        )
        ax[i].text(
            0.0, 1.55, titles[i], transform=ax[i].transAxes, fontsize=20, va="top"
        )

    return f
