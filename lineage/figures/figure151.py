""" This file plots the trees with their predicted states for lapatinib. """

import pickle

from .figureCommon import getSetup
from ..plotTree import plotLineage

# open gemcitabine
pik1 = open("gemcitabines.pkl", "rb")
gemc_tHMMobj_list = []
for _ in range(4):
    gemc_tHMMobj_list.append(pickle.load(pik1))


def makeFigure():
    """
    Makes figure 151.
    """
    ax, f = getSetup((7, 40), (len(gemc_tHMMobj_list[1].X), 1))

    for i in range(len(gemc_tHMMobj_list[1].X)):
        ax[i].axis('off')
        plotLineage(gemc_tHMMobj_list[1].X[i], ax[i])
    return f
