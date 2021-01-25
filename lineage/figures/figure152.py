""" This file plots the trees with their predicted states for lapatinib. """

import numpy as np
from matplotlib.ticker import MaxNLocator
import itertools
import pickle

from .figureCommon import getSetup, subplotLabel
from ..plotTree import plotLineage

# open gemcitabine
pik1 = open("gemcitabines.pkl", "rb")
gemc_tHMMobj_list = []
for _ in range(4):
    gemc_tHMMobj_list.append(pickle.load(pik1))


def makeFigure():
    """
    Makes figure 152.
    """
    ax, f = getSetup((5, 50), (100, 1))
    subplotLabel(ax)

    ax[0].set_title("10 nM Gemcitabine")

    for i in range(100):
        ax[i].axis('off')
        plotLineage(gemc_tHMMobj_list[2].X[i], ax[i])
    return f
