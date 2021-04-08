""" This file plots the trees with their predicted states for lapatinib. """

import pickle

from .figureCommon import getSetup, sort_lins
from ..plotTree import plotLineage

# open gemcitabine
pik1 = open("gemcitabines.pkl", "rb")
gemc_tHMMobj_list = []
for _ in range(4):
    gemc_tHMMobj_list.append(pickle.load(pik1))

gemc_tHMMobj_list[3].X = sort_lins(gemc_tHMMobj_list[3])


def makeFigure():
    """
    Makes figure 153.
    """
    ax, f = getSetup((7, 40), (len(gemc_tHMMobj_list[3].X), 1))

    for i, X in enumerate(gemc_tHMMobj_list[3].X):
        ax[i].axis('off')
        plotLineage(X, ax[i])
    return f
