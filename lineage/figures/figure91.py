""" Plotting the results for HGF. """
""" This file depicts the distribution of phase lengths versus the states for each concentration of lapatinib. """
from string import ascii_lowercase
import pickle
import numpy as np

from .common import getSetup
from ..Lineage_collections import pbs, egf, hgf, osm
from ..plotTree import plot_networkx

HGF = [pbs, egf, hgf, osm]
concs = concsValues = ["PBS", "EGF", "HGF", "OSM"]

pik1 = open("gf.pkl", "rb")
hgf_tHMMobj_list = []
for i in range(4):
    hgf_tHMMobj_list.append(pickle.load(pik1))

T_hgf = hgf_tHMMobj_list[0].estimate.T
num_states = hgf_tHMMobj_list[0].num_states

def makeFigure():
    """ Makes figure 11. """

    ax, f = getSetup((16, 7.5), (2, 6))
    plot2(ax, num_states, hgf_tHMMobj_list, "Growth Factors", concs, concsValues)
    for i in range(3, 6):
        ax[i].set_title(concs[i - 3], fontsize=16)
        ax[i].text(-0.2, 1.25, ascii_lowercase[i-2], transform=ax[i].transAxes, fontsize=16, fontweight="bold", va="top")
        ax[i].axis('off')
    ax[9].set_title(concs[3], fontsize=16)
    ax[9].text(-0.2, 1.25, ascii_lowercase[4], transform=ax[9].transAxes, fontsize=16, fontweight="bold", va="top")
    ax[9].axis('off')
    # plot_networkx(T_hgf.shape[0], T_hgf, 'HGF')
    ax[3].set_title("PBS")
    ax[4].set_title("EGF")
    ax[5].set_title("HGF")
    ax[9].set_title("OSM")

    return f

def plot1(ax, lpt_avg, bern_lpt, cons, concsValues, num_states):
    """ helps to avoid duplicating code for plotting the gamma-related emission results and bernoulli. """
    for i in range(num_states):
        ax[10].plot(cons, lpt_avg[:, i], label="state " + str(i + 1), alpha=0.7)
        ax[10].set_title("Lifetime")
        ax[10].set_ylabel("Log10 Mean Time [hr]")
        ax[10].set_ylim([0.0, 4.0])
        ax[11].plot(cons, bern_lpt[:, i], label="state " + str(i + 1), alpha=0.7)
        ax[11].set_title("Fate")
        ax[11].set_ylabel("Division Probability")
        ax[11].set_ylim([0.0, 1.05])

    # legend and xlabel
    for i in range(10, 12):
        ax[i].legend()
        ax[i].set_xlabel("Growth Factor")
        ax[i].set_xticklabels(concsValues, rotation=30)
        ax[i].text(-0.2, 1.25, ascii_lowercase[i - 5], transform=ax[i].transAxes, fontsize=16, fontweight="bold", va="top")

def plot2(ax, num_states, tHMMobj_list, Dname, cons, concsValues):
    for i in range(3):
        ax[i].axis("off")
        ax[6 + i].axis("off")
    ax[0].text(-0.2, 1.25, ascii_lowercase[0], transform=ax[0].transAxes, fontsize=16, fontweight="bold", va="top")

    # lapatinib
    lpt_avg = np.zeros((4, num_states))  # the avg lifetime: num_conc x num_states x num_phases
    bern_lpt = np.zeros((4, num_states))  # bernoulli

    # print parameters and estimated values
    for idx, tHMMobj in enumerate(tHMMobj_list):  # for each concentration data
        for i in range(num_states):
            lpt_avg[idx, i] = np.log10(tHMMobj.estimate.E[i].params[1] * tHMMobj.estimate.E[i].params[2])
            # bernoullis
            bern_lpt[idx, i] = tHMMobj.estimate.E[i].params[0]

    plot1(ax, lpt_avg, bern_lpt, cons, concsValues, num_states)
