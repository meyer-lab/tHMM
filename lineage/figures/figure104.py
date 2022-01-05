""" This file plots the trees with their predicted states for lapatinib. """

import pickle

from .common import getSetup, sort_lins
from ..plotTree import plotLineage_MCF10A

# open lapatinib
pik1 = open("gf.pkl", "rb")
gf_tHMMobj_list = []
for _ in range(4):
    gf_tHMMobj_list.append(pickle.load(pik1))

gf_tHMMobj_list[0].X = sort_lins(gf_tHMMobj_list[0])


def makeFigure():
    """
    Makes figure 101.
    """
    ax, f = getSetup((7, 15), (len(gf_tHMMobj_list[0].X), 1))

    for i, X in enumerate(gf_tHMMobj_list[0].X):
        ax[i].axis('off')
        plotLineage_MCF10A(X, ax[i])
    return f
