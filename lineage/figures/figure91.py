""" Plotting the results for HGF. """
""" This file depicts the distribution of phase lengths versus the states for each concentration of lapatinib. """
from string import ascii_lowercase
import numpy as np

from .common import getSetup
from ..Analyze import Analyze_list
from ..Lineage_collections import pbs, hgf
from ..plotTree import plot_networkx

HGF = [pbs + hgf]
concs = concsValues = ["PBS", "HGF"]

# HGF
hgf_tHMMobj_list, hgf_states_list, _ = Analyze_list(HGF, 5, fpi=True)

# assign the predicted states to each cell
for idx, hgf_tHMMobj in enumerate(hgf_tHMMobj_list):
    for lin_indx, lin in enumerate(hgf_tHMMobj.X):
        for cell_indx, cell in enumerate(lin.output_lineage):
            cell.state = hgf_states_list[idx][lin_indx][cell_indx]

T_hgf = hgf_tHMMobj_list[0].estimate.T
num_states = hgf_tHMMobj_list[0].num_states


def makeFigure():
    """ Makes figure 11. """

    ax, f = getSetup((15, 6), (2, 4))
    plot2(ax, num_states, hgf_tHMMobj_list, "HGF", concs, concsValues)
    for i in range(2, 4):
        ax[i].set_title(concs[i - 2], fontsize=16)
        ax[i].text(-0.2, 1.25, ascii_lowercase[i - 2], transform=ax[i].transAxes, fontsize=16, fontweight="bold", va="top")
        ax[i].axis('off')
    plot_networkx(5, T_hgf, "HGF")

    return f

def plot1(ax, lpt_avg, bern_lpt, cons, concsValues, num_states):
    """ helps to avoid duplicating code for plotting the gamma-related emission results and bernoulli. """
    for i in range(num_states):  # lapatinib that has 3 states
        ax[6].plot(cons, lpt_avg[:, i], label="state " + str(i + 1), alpha=0.7)
        ax[6].set_title("Lifetime")
        ax[6].set_ylabel("Log10 Mean Time [hr]")
        ax[7].set_ylim([0.0, 30.0])
        ax[7].plot(cons, bern_lpt[:, i], label="state " + str(i + 1), alpha=0.7)
        ax[7].set_title("Fate")
        ax[7].set_ylabel("Division Probability")
        ax[7].set_ylim([0.0, 1.05])

    # legend and xlabel
    for i in range(6, 8):
        ax[i].legend()
        ax[i].set_xlabel("Concentration [nM]")
        ax[i].set_xticklabels(concsValues, rotation=30)
        ax[i].text(-0.2, 1.25, ascii_lowercase[i - 5], transform=ax[i].transAxes, fontsize=16, fontweight="bold", va="top")


def plot2(ax, num_states, tHMMobj_list, Dname, cons, concsValues):
    for i in range(2):
        ax[i].axis("off")
        ax[4 + i].axis("off")
    ax[0].text(-0.2, 1.25, ascii_lowercase[0], transform=ax[0].transAxes, fontsize=16, fontweight="bold", va="top")

    # lapatinib
    lpt_avg = np.zeros((2, num_states))  # the avg lifetime: num_conc x num_states x num_phases
    bern_lpt = np.zeros((2, num_states))  # bernoulli
    # print parameters and estimated values
    print(Dname, "\n the \u03C0: ", tHMMobj_list[0].estimate.pi, "\n the transition matrix: ", tHMMobj_list[0].estimate.T)
    for idx, tHMMobj in enumerate(tHMMobj_list):  # for each concentration data
        for i in range(num_states):
            lpt_avg[idx, i] = tHMMobj.estimate.E[i].params[1] * tHMMobj.estimate.E[i].params[2]
            # bernoullis
            bern_lpt[idx, i] = tHMMobj.estimate.E[i].params[0]

    plot1(ax, lpt_avg, bern_lpt, cons, concsValues, num_states)
