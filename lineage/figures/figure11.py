""" This file depicts the distribution of phase lengths versus the states for each concentration. """
import numpy as np
import itertools
import seaborn as sns

from ..Analyze import Analyze_list
from ..tHMM import tHMM
from ..data.Lineage_collections import gemControl, gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM
from .figureCommon import getSetup, subplotLabel
# np.random.seed(1)

concs = ["cntrl", "Lapt 25nM", "Lapt 50nM", "Lapt 250nM", "cntrl", "Gem 5nM", "Gem 10nM", "Gem 30nM"]
concsValues = ["cntrl", "25nM", "50nM", "250nM", "cntrl", "5nM", "10nM", "30nM"]
data = [Lapatinib_Control + gemControl, Lapt25uM, Lapt50uM, Lap250uM, gemControl + Lapatinib_Control, gem5uM, Gem10uM, Gem30uM]

tHMM_solver = tHMM(X=data[0], num_states=1)
tHMM_solver.fit()

constant_shape = [int(tHMM_solver.estimate.E[0].params[2]), int(tHMM_solver.estimate.E[0].params[4])]

# Set shape
for population in data:
    for lin in population:
        for E in lin.E:
            E.G1.const_shape = constant_shape[0]
            E.G2.const_shape = constant_shape[1]

# Run fitting
lapt_tHMMobj_list, lapt_states_list, _ = Analyze_list(data[0:4], 3, fpi=True)
gemc_tHMMobj_list, gemc_states_list, _ = Analyze_list(data[4:], 4, fpi=True)


def makeFigure():
    """ Makes figure 11. """

    ax, f = getSetup((13.2, 10.0), (4, 4))

    # lapatinib
    lpt_avg = np.zeros((4, 3, 2))  # the avg lifetime: num_conc x num_states x num_phases
    bern_lpt = np.zeros((4, 3, 2))  # bernoulli
    # print parameters and estimated values
    print("for Lapatinib: \n the \u03C0: ", lapt_tHMMobj_list[0].estimate.pi, "\n the transition matrix: ", lapt_tHMMobj_list[0].estimate.T)

    for idx, lapt_tHMMobj in enumerate(lapt_tHMMobj_list):  # for each concentration data
        for i in range(3):
            lpt_avg[idx, i, 0] = lapt_tHMMobj.estimate.E[i].params[2] * lapt_tHMMobj.estimate.E[i].params[3]  # G1
            lpt_avg[idx, i, 1] = lapt_tHMMobj.estimate.E[i].params[4] * lapt_tHMMobj.estimate.E[i].params[5]  # G2
            # bernoullis
            for j in range(2):
                bern_lpt[idx, i, j] = lapt_tHMMobj.estimate.E[i].params[j]

        LAP_state, LAP_phaseLength, Lpt_phase = twice(lapt_tHMMobj, lapt_states_list[idx])

        # plot lapatinib
        sns.stripplot(x=LAP_state, y=LAP_phaseLength, hue=Lpt_phase, size=1.5, palette="Set2", dodge=True, ax=ax[idx])

        ax[idx].set_title(concs[idx])
        ax[idx + 4].set_title(concs[idx + 4])
        ax[idx].set_ylabel("phase lengths")
        ax[idx + 4].set_ylabel("phase lengths")
        ax[idx].set_xlabel("state")
        ax[idx + 4].set_xlabel("state")
        ax[idx].set_ylim([0, 200])
        ax[idx + 4].set_ylim([0, 200])

    # gemcitabine
    gmc_avg = np.zeros((4, 4, 2))  # avg lifetime gmc: num_conc x num_states x num_phases
    bern_gmc = np.zeros((4, 4, 2))  # bernoulli
    # print parameters and estimated values
    print("for Gemcitabine: \n the \u03C0: ", gemc_tHMMobj_list[0].estimate.pi, " \n the transition matrix: ", gemc_tHMMobj_list[0].estimate.T)

    for idx, gemc_tHMMobj in enumerate(gemc_tHMMobj_list):
        for i in range(4):
            gmc_avg[idx, i, 0] = gemc_tHMMobj.estimate.E[i].params[2] * gemc_tHMMobj.estimate.E[i].params[3]
            gmc_avg[idx, i, 1] = gemc_tHMMobj.estimate.E[i].params[4] * gemc_tHMMobj.estimate.E[i].params[5]
            # bernoulli
            for j in range(2):
                bern_gmc[idx, i, j] = gemc_tHMMobj.estimate.E[i].params[j]
        GEM_state, GEM_phaseLength, GEM_phase = twice(gemc_tHMMobj, gemc_states_list[idx])
        sns.stripplot(x=GEM_state, y=GEM_phaseLength, hue=GEM_phase, size=1.5, palette="Set2", dodge=True, ax=ax[idx + 4])

    plotting(ax, 8, lpt_avg, gmc_avg, concs, "avg lengths")
    plotting(ax, 12, bern_lpt, bern_gmc, concs, "Bernoulli p")
    return f


def plotting(ax, k, lpt_avg, gmc_avg, concs, title):
    """ helps to avoid duplicating code for plotting the gamma-related emission results and bernoulli. """
    for i in range(3):  # lapatinib that has 3 states
        ax[k].plot(concs[0:4], lpt_avg[:, i, 0], label="st " + str(i), alpha=0.7)
        ax[k].set_title(title + str(" G1"))
        ax[k].set_xticklabels(concsValues[0:4], rotation=30)
        ax[k + 1].plot(concs[0:4], lpt_avg[:, i, 1], label="st " + str(i), alpha=0.7)
        ax[k + 1].set_title(title + str(" G2"))
        ax[k + 1].set_xticklabels(concsValues[0:4], rotation=30)

    for i in range(4):  # gemcitabine that has 4 states
        ax[k + 2].plot(concs[4:8], gmc_avg[:, i, 0], label="st " + str(i), alpha=0.7)
        ax[k + 2].set_title(title + str(" G1"))
        ax[k + 2].set_xticklabels(concsValues[4:8], rotation=30)
        ax[k + 3].plot(concs[4:8], gmc_avg[:, i, 1], label="st " + str(i), alpha=0.7)
        ax[k + 3].set_title(title + str(" G2"))
        ax[k + 3].set_xticklabels(concsValues[4:8], rotation=30)

    # legend and ylabel
    for i in range(k, k + 4):
        ax[i].legend()
        ax[i].set_ylabel("phase duration")

    #lapatinib xlabel
    for i in range(k, k + 2):
        ax[i].set_xlabel("lapatinib")

    #gemcitibine xlabel
    for i in range(k + 2, k + 4):
        ax[i].set_xlabel("gemcitabine")

    # ylim for lapatinib
    if k == 8:
        for i in range(k, k + 4):
            ax[i].set_ylim([0, 200])
    elif k == 12:
        for i in range(k, k + 4):
            ax[i].set_ylim([0, 1.05])

    subplotLabel(ax)


def twice(tHMMobj, state):
    """ For each tHMM object, connects the state and the emissions. """
    g1 = []
    g2 = []
    for lin in tHMMobj.X:  # for each lineage list
        for cell in lin.output_lineage:  # for each cell in the lineage
            if cell.obs[4] == 1:
                g1.append(cell.obs[2])
            else:
                g1.append(np.nan)
            if cell.obs[5] == 1:
                g2.append(cell.obs[3])
            else:
                g2.append(np.nan)

    state = list(itertools.chain(*state)) + list(itertools.chain(*state))
    phaseLength = g1 + g2
    phase = len(g1) * ["G1"] + len(g2) * ["G2"]
    return state, phaseLength, phase
