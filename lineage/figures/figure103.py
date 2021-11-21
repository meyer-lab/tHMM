""" This file plots the trees with their predicted states for lapatinib. """

import pickle

from .common import getSetup, sort_lins
from ..plotTree import plotLineage

# open lapatinib
pik1 = open("lapatinibs.pkl", "rb")
lapt_tHMMobj_list = []
for _ in range(4):
    lapt_tHMMobj_list.append(pickle.load(pik1))

lapt_tHMMobj_list[3].X = sort_lins(lapt_tHMMobj_list[3])


def makeFigure():
    """
    Makes figure 103.
    """
    ax, f = getSetup((7, 40), (len(lapt_tHMMobj_list[3].X), 1))

    for i, X in enumerate(lapt_tHMMobj_list[3].X):
        ax[i].axis('off')
        plotLineage(X, ax[i])
    return f
