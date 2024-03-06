""" This file plots the trees with their predicted states for lapatinib. """

from .common import getSetup
from ..plotTree import plotLineage
from ..Analyze import Analyze_list
from ..Lineage_collections import AllGemcitabine

num_states = 5
gemc_tHMMobj_list = Analyze_list(AllGemcitabine, num_states, write_states=True)[0]

only_lapatinib_control_1 = gemc_tHMMobj_list[0].X[0:100]


def makeFigure():
    """
    Makes figure 161.
    """
    ax, f = getSetup((4, 40), (60, 1))

    for i in range(60):
        ax[i].axis("off")
        plotLineage(only_lapatinib_control_1[i], ax[i])

    return f
