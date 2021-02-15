""" This file plots the trees with their predicted states for lapatinib. """

import pickle

from .figureCommon import getSetup
from ..plotTree import plotLineage

# open lapatinib
pik1 = open("lapatinibs.pkl", "rb")
lapt_tHMMobj_list = []
for _ in range(4):
    lapt_tHMMobj_list.append(pickle.load(pik1))

only_lapatinib_control_1 = lapt_tHMMobj_list[0].X[0:100]


def makeFigure():
    """
    Makes figure 16.
    """
    ax, f = getSetup((4, 40), (60, 1))

    for i in range(60):
        ax[i].axis('off')
        plotLineage(only_lapatinib_control_1[i], ax[i])

    return f
