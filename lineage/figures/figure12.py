""" This file depicts the distribution of phase lengths versus the states for each concentration of gemcitabine. """
import pickle

from .figureCommon import getSetup, subplotLabel, plot_all
from ..plotTree import plot_networkx


concs = ["control", "gemcitabine 5 nM", "gemcitabine 10 nM", "gemcitabine 30 nM"]
concsValues = ["control", "5 nM", "10 nM", "30 nM"]

pik1 = open("gemcitabines.pkl", "rb")
gemc_tHMMobj_list = []
for i in range(4):
    gemc_tHMMobj_list.append(pickle.load(pik1))

T_gem = gemc_tHMMobj_list[0].estimate.T
num_states = gemc_tHMMobj_list[0].num_states


def makeFigure():
    """ Makes figure 12. """

    ax, f = getSetup((22, 7.0), (2, 6))
    for u in range(4, 6):
        ax[u].axis("off")
    for u in range(10, 12):
        ax[u].axis("off")
    plot_all(ax, num_states, gemc_tHMMobj_list, "Gemcitabine", concs, concsValues)
    return f


plot_networkx(T_gem.shape[0], T_gem, 'gemcitabine')
