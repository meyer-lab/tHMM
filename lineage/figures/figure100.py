""" This file plots the trees with their predicted states for lapatinib. """

import pickle

from .common import getSetup, sort_lins
from ..plotTree import plotLineage

# open gemcitabine
pik1 = open("lapatinibs.pkl", "rb")
lapt_tHMMobj_list = []
for _ in range(4):
    lapt_tHMMobj_list.append(pickle.load(pik1))

lapt_tHMMobj_list[0].X = sort_lins(lapt_tHMMobj_list[0])


def makeFigure():
    """
    Makes figure 150.
    """
    ax, f = getSetup((7, 40), (len(lapt_tHMMobj_list[0].X), 1))

    for i, X in enumerate(lapt_tHMMobj_list[0].X):
        ax[i].axis('off')
        plotLineage(X, ax[i])
    return f
