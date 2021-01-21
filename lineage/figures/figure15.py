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
    Makes figure 10.
    """
    ax, f = getSetup((4, 40), (50, 4))
    subplotLabel(ax)

    # Plotting the lineages
    figure_maker(ax, gemc_tHMMobj_list)
    return f


def figure_maker(ax, gemcitabine):
    """
    Makes figure 10.
    """

    ax[0].set_title("Control")
    ax[1].set_title("5 nM")
    ax[2].set_title("10 nM")
    ax[3].set_title("30 nM")

    j = 0
    for i in range(50):
        ax[j].axis('off')
        plotLineage(gemcitabine[0].X[i], ax[j])
        ax[j+1].axis('off')
        plotLineage(gemcitabine[1].X[i], ax[j+1])
        ax[j+2].axis('off')
        plotLineage(gemcitabine[2].X[i], ax[j+2])
        ax[j+3].axis('off')
        plotLineage(gemcitabine[3].X[i], ax[j+3])
        j += 4
