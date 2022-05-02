""" This file depicts the distribution of phase lengths versus the states for each concentration of gemcitabine. """
import pickle
from string import ascii_lowercase
from ..plotTree import plot_networkx
from .common import getSetup, plot_all, subplotLabel

concs = ["Control", "Gemcitabine 5 nM", "Gemcitabine 10 nM", "Gemcitabine 30 nM"]
concsValues = ["Control", "5 nM", "10 nM", "30 nM"]

pik1 = open("gemcitabines.pkl", "rb")
gemc_tHMMobj_list = []
for i in range(4):
    gemc_tHMMobj_list.append(pickle.load(pik1))

num_states = gemc_tHMMobj_list[0].num_states


def makeFigure():
    """ Makes figure 12. """
    ax, f = getSetup((12, 6), (3, 4))
    plot_all(ax, num_states, gemc_tHMMobj_list, "Gemcitabine", concs, concsValues)
    subplotLabel(ax)

    for i in range(8):
        ax[i].axis("off")

    return f

# plot_networkx(num_states, gemc_tHMMobj_list[0].estimate.T, "Gemcitabine")