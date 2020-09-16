""" This file depicts the distribution of phase lengths versus the states. """

import numpy as np
import seaborn as sns

from ..data.Lineage_collections import Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM
from .figureCommon import getSetup, subplotLabel
from .figure11 import lapatinib12, gemcitabine12


def makeFigure():
    """
    Makes figure 12.
    """
    ax, f = getSetup((10, 10), (4, 4))
    subplotLabel(ax)

    concs = ["cntrl", "Lapt 25uM", "Lapt 50uM", "Lapt 250uM", "cntrl", "Gem 5uM", "Gem 10uM", "Gem 30uM"]
    Lap = pd.DataFrame(columns=["state","cntrl G1", "Lapt 25uM G1", "Lapt 50uM G1", "Lapt 250uM G1", "cntrl G2", "Lapt 25uM G2", "Lapt 50uM G2", "Lapt 250uM G2"])
    Gem = pd.DataFrame(columns=["state", "cntrl G1", "Gem 5uM G1", "Gem 10uM G1", "Gem 30uM G1", "cntrl G2", "Gem 5uM G2", "Gem 10uM G2", "Gem 30uM G2"])

    # create the dataframe
    for i in range(4):
        Lap["state"] = [a[0] for a in lapatinib12[i]]
        Lap[str(concs[i])+" G1"] = [a[1] for a in lapatinib12[i]]
        Lap[str(concs[i])+" G2"] = [a[2] for a in lapatinib12[i]]
        Gem["state"] = [a[0] for a in gemcitabine12[i]]
        Gem[str(concs[i+4])+" G1"] = [a[1] for a in gemcitabine12[i]]
        Gem[str(concs[i+4])+" G2"] = [a[2] for a in gemcitabine12[i]]

    # plot lapatinib
    for i in range(4): # G1 lapatinib (first row), lapatinib G2 (second row)
        sns.stripplot(x="state", y=str(concs[i]+" G1"), data=Lap, ax=ax[i], linewidth=1, jitter=0.1)
        sns.stripplot(x="state", y=str(concs[i]+" G2"), data=Lap, ax=ax[i+4], linewidth=1, jitter=0.1)
        ax[i].set_title(concs[i])
        ax[i].set_ylabel("G1 phase lengths")
        ax[i+4].set_ylabel("G2 phase lengths")

    # plot gemcitabine
    for i in range(8, 12):
        sns.stripplot(x="state", y=str(concs[i-4]+" G1"), data=Gem, ax=ax[i], linewidth=1, jitter=0.1)
        sns.stripplot(x="state", y=str(concs[i-4]+" G2"), data=Gem, ax=ax[i+4], linewidth=1, jitter=0.1)
        ax[i].set_title(concs[i])
        ax[i].set_ylabel("G1 phase lengths")
        ax[i+4].set_ylabel("G2 phase lengths")

    return f
