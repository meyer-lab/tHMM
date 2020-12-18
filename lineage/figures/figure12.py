""" This file depicts the distribution of phase lengths versus the states for each concentration of gemcitabine. """
import numpy as np
import itertools
import seaborn as sns
from string import ascii_lowercase


from ..Analyze import Analyze_list
from ..tHMM import tHMM
from ..data.Lineage_collections import gemControl, gem5uM, Gem10uM, Gem30uM, Lapatinib_Control
from .figureCommon import getSetup, subplotLabel
from .figure11 import twice, plot_networkx


data = [gemControl + Lapatinib_Control, gem5uM, Gem10uM, Gem30uM]
concs = ["cntrl", "Gem 5nM", "Gem 10nM", "Gem 30nM"]
concsValues = ["cntrl", "5nM", "10nM", "30nM"]


tHMM_solver = tHMM(X=data[0], num_states=1)
tHMM_solver.fit()

num_states = 3
gemc_tHMMobj_list, gemc_states_list, _ = Analyze_list(data, num_states, fpi=True)
T_gem = gemc_tHMMobj_list[0].estimate.T


def makeFigure():
    """ Makes figure 12. """

    ax, f = getSetup((16, 6.0), (2, 5))
    ax[4].axis("off")
    ax[9].axis("off")
    ax[4].text(-0.2, 1.25, ascii_lowercase[8], transform=ax[4].transAxes, fontsize=16, fontweight="bold", va="top")

    # gemcitabine
    gmc_avg = np.zeros((4, num_states, 2))  # avg lifetime gmc: num_conc x num_states x num_phases
    bern_gmc = np.zeros((4, num_states, 2))  # bernoulli
    # print parameters and estimated values
    print("for Gemcitabine: \n the \u03C0: ", gemc_tHMMobj_list[0].estimate.pi, " \n the transition matrix: ", gemc_tHMMobj_list[0].estimate.T)

    for idx, gemc_tHMMobj in enumerate(gemc_tHMMobj_list):
        for i in range(num_states):
            gmc_avg[idx, i, 0] = 1 / (gemc_tHMMobj.estimate.E[i].params[2] * gemc_tHMMobj.estimate.E[i].params[3])
            gmc_avg[idx, i, 1] = 1 / (gemc_tHMMobj.estimate.E[i].params[4] * gemc_tHMMobj.estimate.E[i].params[5])
            # bernoulli
            for j in range(2):
                bern_gmc[idx, i, j] = gemc_tHMMobj.estimate.E[i].params[j]

        gemc_states_plusone = [i + 1 for i in gemc_states_list[idx]]
        GEM_state, GEM_phaseLength, GEM_phase = twice(gemc_tHMMobj, gemc_states_plusone)
        sns.stripplot(x=GEM_state, y=GEM_phaseLength, hue=GEM_phase, size=1.5, palette="Set2", dodge=True, ax=ax[idx])

        ax[idx].set_title(concs[idx])
        ax[idx].set_ylabel("phase lengths [hr]")
        ax[idx].set_xlabel("state")
        ax[idx].set_ylim([0.0, 150.0])

    plot_gemc(ax, gmc_avg, bern_gmc, concs)
    return f


def plot_gemc(ax, gmc_avg, bern_gmc, concs):

    for i in range(num_states):  # gemcitabine that has 3 states
        ax[5].plot(concs, gmc_avg[:, i, 0], label="st " + str(i + 1), alpha=0.7)
        ax[5].set_title("G1 phase")
        ax[6].plot(concs, gmc_avg[:, i, 1], label="st " + str(i + 1), alpha=0.7)
        ax[6].set_title("G2 phase")
        ax[7].plot(concs, bern_gmc[:, i, 0], label="st " + str(i + 1), alpha=0.7)
        ax[7].set_title("G1 phase")
        ax[8].plot(concs, bern_gmc[:, i, 1], label="st " + str(i + 1), alpha=0.7)
        ax[8].set_title("G2 phase")

    # ylim and ylabel
    for i in range(5, 7):
        ax[i].set_ylabel("prog. rate 1/[hr]")
        ax[i].set_ylim([0, 0.05])

    # ylim and ylabel
    for i in range(7, 9):
        ax[i].set_ylabel("death rate")
        ax[i].set_ylim([0, 1.05])

    # legend and xlabel
    for i in range(5, 9):
        ax[i].legend()
        ax[i].set_xlabel("conc. [nM]")
        ax[i].set_xticklabels(concsValues, rotation=30)

    subplotLabel(ax)


plot_networkx(T_gem.shape[0], T_gem, 'gemcitabine')
