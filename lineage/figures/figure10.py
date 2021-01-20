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

# open gemcitabine
pik1 = open("gemcitabines.pkl", "rb")
gemc_tHMMobj_list = []
for _ in range(4):
    gemc_tHMMobj_list.append(pickle.load(pik1))

length0 = len(lapt_tHMMobj_list[0].X) # control
length1 = len(lapt_tHMMobj_list[1].X) # 25 nM
length2 = len(lapt_tHMMobj_list[2].X) # 50 nM
length3 = len(lapt_tHMMobj_list[3].X) # 250 nM
lng = np.min(length0, length1, length2, length3)
def makeFigure():
    """
    Makes figure 10.
    """
    ax, f = getSetup((20, 50), (lng, 4))
    subplotLabel(ax)

    # Plotting the lineages
    figure_maker(ax, lapt_tHMMobj_list)

    return f


def figure_maker(ax, lapatinib):
    """
    Makes figure 10.
    """

    ax[0].set_title("Control")
    ax[1].set_title("25 nM")
    ax[2].set_title("50 nM")
    ax[3].set_title("250 nM")

    j = 0
    for i in range(lng):
        ax[j].axis('off')
        plotLineage(lapatinib[0].X[i], ax[j])
        ax[j+1].axis('off')
        plotLineage(lapatinib[1].X[i], ax[j+1])
        ax[j+2].axis('off')
        plotLineage(lapatinib[2].X[i], ax[j+2])
        ax[j+3].axis('off')
        plotLineage(lapatinib[3].X[i], ax[j+3])
        j += 4
