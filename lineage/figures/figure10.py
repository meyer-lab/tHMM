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

length = len(lapt_tHMMobj_list[1].X) # 25 nM

def makeFigure():
    """
    Makes figure 10.
    """
    ax, f = getSetup((50, 5), (length, 1))
    subplotLabel(ax)

    # Plotting the lineages
    figure_maker(ax, list(itertools.chain(*lapt_tHMMobj_list[1].X)))

    return f


def figure_maker(ax, lapatinib):
    """
    Makes figure 10.
    """

    ax[0].set_title("Lapatinib")

    for j in range(length):
        ax[j].axis('off')
        plotLineage(lapatinib[j], ax[j])
