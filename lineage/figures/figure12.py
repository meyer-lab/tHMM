""" This file depicts the distribution of phase lengths versus the states for each concentration of gemcitabine. """
import numpy as np
import itertools
import seaborn as sns

from ..Analyze import Analyze_list
from ..tHMM import tHMM
from ..data.Lineage_collections import gemControl, gem5uM, Gem10uM, Gem30uM, Lapatinib_Control
from .figureCommon import getSetup, subplotLabel
from .figure11 import twice
np.random.seed(1)


data = [gemControl + Lapatinib_Control, gem5uM, Gem10uM, Gem30uM]
concs = ["cntrl", "Gem 5nM", "Gem 10nM", "Gem 30nM"]
concsValues = ["cntrl", "5nM", "10nM", "30nM"]


tHMM_solver = tHMM(X=data[0], num_states=1)
tHMM_solver.fit()

constant_shape = [int(tHMM_solver.estimate.E[0].params[2]), int(tHMM_solver.estimate.E[0].params[4])]

# Set shape
for population in data:
    for lin in population:
        for E in lin.E:
            E.G1.const_shape = constant_shape[0]
            E.G2.const_shape = constant_shape[1]

gemc_tHMMobj_list, gemc_states_list, _ = Analyze_list(data, 4, fpi=True)


def makeFigure():
    """ Makes figure 12. """

    ax, f = getSetup((13.2, 6.0), (2, 5))
    ax[4].axis("off")
    ax[9].axis("off")

    # gemcitabine
    gmc_avg = np.zeros((4, 4, 2))  # avg lifetime gmc: num_conc x num_states x num_phases
    bern_gmc = np.zeros((4, 4, 2))  # bernoulli
    # print parameters and estimated values
    print("for Gemcitabine: \n the \u03C0: ", gemc_tHMMobj_list[0].estimate.pi, " \n the transition matrix: ", gemc_tHMMobj_list[0].estimate.T)

    for idx, gemc_tHMMobj in enumerate(gemc_tHMMobj_list):
        for i in range(4):
            gmc_avg[idx, i, 0] = 1 / (gemc_tHMMobj.estimate.E[i].params[2] * gemc_tHMMobj.estimate.E[i].params[3])
            gmc_avg[idx, i, 1] = 1 / (gemc_tHMMobj.estimate.E[i].params[4] * gemc_tHMMobj.estimate.E[i].params[5])
            # bernoulli
            for j in range(2):
                bern_gmc[idx, i, j] = gemc_tHMMobj.estimate.E[i].params[j]

        GEM_state, GEM_phaseLength, GEM_phase = twice(gemc_tHMMobj, gemc_states_list[idx])
        sns.stripplot(x=GEM_state, y=GEM_phaseLength, hue=GEM_phase, size=1.5, palette="Set2", dodge=True, ax=ax[idx])

        ax[idx].set_title(concs[idx])
        ax[idx].set_ylabel("phase lengths")
        ax[idx].set_xlabel("state")
        ax[idx].set_ylim([0.0, 150.0])

    plot_gemc(ax, gmc_avg, bern_gmc, concs)
    return f

def plot_gemc(ax, gmc_avg, bern_gmc, concs):

    for i in range(4):  # gemcitabine that has 4 states
        ax[5].plot(concs, gmc_avg[:, i, 0], label="st " + str(i), alpha=0.7)
        ax[6].plot(concs, gmc_avg[:, i, 1], label="st " + str(i), alpha=0.7)
        ax[7].plot(concs, bern_gmc[:, i, 0], label="st " + str(i), alpha=0.7)
        ax[8].plot(concs, bern_gmc[:, i, 1], label="st " + str(i), alpha=0.7)


    # legend and ylabel
    for i in range(5, 7):
        ax[i].legend()
        ax[i].set_xticklabels(concsValues, rotation=30)
        ax[i].set_title("G1 phase")
        ax[i].set_ylabel("prog. rate")
        ax[i].set_ylim([0, 0.1])

    # xlabel
    for i in range(7, 9):
        ax[i].legend()
        ax[i].set_xticklabels(concsValues, rotation=30)
        ax[i].set_title("G2 phase")
        ax[i].set_ylabel("div. rate")
        ax[i].set_ylim([0, 1.05])

    subplotLabel(ax)
