""" This file plots the trees with their predicted states for lapatinib. """

import pickle

from .common import getSetup
from ..plotTree import plotLineage

# open gemcitabine
pik1 = open("gemcitabines.pkl", "rb")
alls = []
for i in range(7):
    gemc_tHMMobj_list = []
    for i in range(4):
        gemc_tHMMobj_list.append(pickle.load(pik1))
    alls.append(gemc_tHMMobj_list)

# selected for gemcitabine is 5 states which is index 4.
gemc_tHMMobj_list = alls[4]

only_lapatinib_control_1 = gemc_tHMMobj_list[0].X[0:100]


def makeFigure():
    """
    Makes figure 161.
    """
    ax, f = getSetup((4, 40), (60, 1))

    for i in range(60):
        ax[i].axis('off')
        plotLineage(only_lapatinib_control_1[i], ax[i])

    return f
