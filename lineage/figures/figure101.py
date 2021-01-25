""" This file plots the trees with their predicted states for lapatinib. """

import numpy as np
from matplotlib.ticker import MaxNLocator
import itertools
import pickle

from .figureCommon import getSetup, subplotLabel
from ..plotTree import plotLineage

# open lapatinib
pik1 = open("lapatinibs.pkl", "rb")
lapt_tHMMobj_list = []
for _ in range(4):
    lapt_tHMMobj_list.append(pickle.load(pik1))


def makeFigure():
    """
    Makes figure 101.
    """
    ax, f = getSetup((5, 50), (100, 1))
    subplotLabel(ax)

    ax[0].set_title("25 nM Lapatinib")

    for i in range(100):
        ax[i].axis('off')
        plotLineage(lapt_tHMMobj_list[1].X[i], ax[i])
    return f
