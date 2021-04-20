""" This file depicts the distribution of phase lengths versus the states for each concentration of lapatinib. """
import pickle
from string import ascii_lowercase

from .figureCommon import getSetup, plot_all
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

    ax, f = getSetup((17, 7.5), (2, 7))
    plot_all(ax, num_states, lapt_tHMMobj_list, "Laptinib", concs, concsValues)
    for i in range(3, 7):
        ax[i].set_title(concs[i-3], fontsize=16)
        ax[i].text(-0.2, 1.25, ascii_lowercase[i-2], transform=ax[i].transAxes, fontsize=16, fontweight="bold", va="top")
        ax[i].axis('off')

    return f

# plot_networkx(T_lap.shape[0], T_lap, 'lapatinib')
