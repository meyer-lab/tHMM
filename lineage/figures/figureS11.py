""" This is a file to put together 4 conditions of lapatinib together. """

from string import ascii_lowercase
from .common import getSetup, sort_lins
from ..plotTree import plotLineage
from ..Analyze import Analyze_list
from ..Lineage_collections import AllLapatinib

num_states = 4
lapt_tHMMobj_list = Analyze_list(AllLapatinib, num_states)[0]

for i in range(4):
    lapt_tHMMobj_list[i].X = sort_lins(lapt_tHMMobj_list[i])


def makeFigure():
    """
    Makes figure 100.
    """
    titles = ["Control", "Lapatinib 25 nM", "Lapatinib 50 nM", "Lapatinib 250 nM"]
    ax, f = getSetup((15, 45), (170, 4))

    for i in range(170):
        ax[4 * i].axis("off")
        ax[4 * i + 1].axis("off")
        ax[4 * i + 2].axis("off")
        ax[4 * i + 3].axis("off")
        plotLineage(lapt_tHMMobj_list[0].X[i], ax[4 * i])
        plotLineage(lapt_tHMMobj_list[1].X[i], ax[4 * i + 1])
        plotLineage(lapt_tHMMobj_list[2].X[i], ax[4 * i + 2])
        plotLineage(lapt_tHMMobj_list[3].X[i], ax[4 * i + 3])

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
