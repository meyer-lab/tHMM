""" This file plots the trees with their predicted states for lapatinib. """

import pickle

from .figureCommon import getSetup
from ..plotTree import plotLineage

# open lapatinib
pik1 = open("lapatinibs.pkl", "rb")
lapt_tHMMobj_list = []
for _ in range(4):
    lapt_tHMMobj_list.append(pickle.load(pik1))


def makeFigure():
    """
    Makes figure 10.
    """
    ax, f = getSetup((7, 40), (100, 1))

    for i in range(100):
        ax[i].axis('off')
        plotLineage(lapt_tHMMobj_list[0].X[i], ax[i])
    return f
