""" This file depicts the distribution of phase lengths versus the states for each concentration of lapatinib. """
import pickle
from string import ascii_lowercase
from ..plotTree import plot_networkx
from .common import getSetup, plot_all, subplotLabel

concs = ["Control", "Lapatinib 25 nM", "Lapatinib 50 nM", "Lapatinib 250 nM"]
concsValues = ["Control", "25 nM", "50 nM", "250 nM"]

pik1 = open("lapatinibs.pkl", "rb")
lapt_tHMMobj_list = []
for i in range(4):
    lapt_tHMMobj_list.append(pickle.load(pik1))

num_states = lapt_tHMMobj_list[0].num_states


def makeFigure():
    """ Makes figure 11. """

    ax, f = getSetup((12, 6), (3, 4))
    subplotLabel(ax)
    for i in range(8):
        ax[i].axis("off")

    plot_all(ax, num_states, lapt_tHMMobj_list, "Laptinib", concs, concsValues)

    return f
