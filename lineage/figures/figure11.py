""" This file depicts the distribution of phase lengths versus the states for each concentration of lapatinib. """
import numpy as np
import seaborn as sns
from string import ascii_lowercase
import pickle

from ..tHMM import tHMM
from .figureCommon import getSetup, subplotLabel, plot_all
from ..plotTree import plot_networkx

concs = ["control", "lapatinib 25 nM", "lapatinib 50 nM", "lapatinib 250 nM"]
concsValues = ["control", "25 nM", "50 nM", "250 nM"]

pik1 = open("lapatinibs.pkl", "rb")
lapt_tHMMobj_list = []
for i in range(4):
    lapt_tHMMobj_list.append(pickle.load(pik1))

T_lap = lapt_tHMMobj_list[0].estimate.T
num_states = lapt_tHMMobj_list[0].num_states


def makeFigure():
    """ Makes figure 11. """

    ax, f = getSetup((22, 7.0), (2, 6))
    for u in range(4, 6):
        ax[u].axis("off")
    for u in range(10, 12):
        ax[u].axis("off")

    plot_all(ax, num_states, lapt_tHMMobj_list, "Laptinib", concs, concsValues)
    return f


plot_networkx(T_lap.shape[0], T_lap, 'lapatinib')
