""" This file depicts the distribution of phase lengths versus the states for each concentration of lapatinib. """
import numpy as np
import itertools
import seaborn as sns

from ..Analyze import Analyze_list
from ..tHMM import tHMM
from ..data.Lineage_collections import gemControl, gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM
from .figureCommon import getSetup, subplotLabel
np.random.seed(1)

concs = ["cntrl", "Lapt 25nM", "Lapt 50nM", "Lapt 250nM", "cntrl", "5nM", "10nM", "30nM"]
concsValues = ["cntrl", "25nM", "50nM", "250nM"]
data = [Lapatinib_Control + gemControl, Lapt25uM, Lapt50uM, Lap250uM]

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
lapt_tHMMobj_list, lapt_states_list, _ = Analyze_list(data, 3, fpi=True)


def makeFigure():
    """ Makes figure 11. """

    ax, f = getSetup((13.2, 6.0), (2, 5))
    ax[4].axis("off")
    ax[9].axis("off")

    # lapatinib
    lpt_avg = np.zeros((4, 3, 2))  # the avg lifetime: num_conc x num_states x num_phases
    bern_lpt = np.zeros((4, 3, 2))  # bernoulli
    # print parameters and estimated values
    print("for Lapatinib: \n the \u03C0: ", lapt_tHMMobj_list[0].estimate.pi, "\n the transition matrix: ", lapt_tHMMobj_list[0].estimate.T)

    for idx, lapt_tHMMobj in enumerate(lapt_tHMMobj_list):  # for each concentration data
        for i in range(3):
            lpt_avg[idx, i, 0] = 1 / (lapt_tHMMobj.estimate.E[i].params[2] * lapt_tHMMobj.estimate.E[i].params[3])  # G1
            lpt_avg[idx, i, 1] = 1 / (lapt_tHMMobj.estimate.E[i].params[4] * lapt_tHMMobj.estimate.E[i].params[5])  # G2
            # bernoullis
            for j in range(2):
                bern_lpt[idx, i, j] = lapt_tHMMobj.estimate.E[i].params[j]

        LAP_state, LAP_phaseLength, Lpt_phase = twice(lapt_tHMMobj, lapt_states_list[idx])

        # plot lapatinib
        sns.stripplot(x=LAP_state, y=LAP_phaseLength, hue=Lpt_phase, size=1.5, palette="Set2", dodge=True, ax=ax[idx])

        ax[idx].set_title(concs[idx])
        ax[idx].set_ylabel("phase lengths [hr]")
        ax[idx].set_xlabel("state")
        ax[idx].set_ylim([0.0, 150.0])

    plotting(ax, lpt_avg, bern_lpt, concs)
    return f


def plotting(ax, lpt_avg, bern_lpt, concs):
    """ helps to avoid duplicating code for plotting the gamma-related emission results and bernoulli. """
    for i in range(3):  # lapatinib that has 3 states
        ax[5].plot(concs[0: 4], lpt_avg[:, i, 0], label="st " + str(i), alpha=0.7)
        ax[5].set_title("G1 phase")
        ax[6].plot(concs[0: 4], lpt_avg[:, i, 1], label="st " + str(i), alpha=0.7)
        ax[6].set_title("G2 phase")
        ax[7].plot(concs[0: 4], bern_lpt[:, i, 0], label="st " + str(i), alpha=0.7)
        ax[7].set_title("G1 phase")
        ax[8].plot(concs[0: 4], bern_lpt[:, i, 1], label="st " + str(i), alpha=0.7)
        ax[8].set_title("G2 phase")

    # ylim and ylabel
    for i in range(5, 7):
        ax[i].set_ylabel("prog. rate 1/[hr]")
        ax[i].set_ylim([0, 0.1])

    # ylim and ylabel
    for i in range(7, 9):
        ax[i].set_ylabel("div. rate")
        ax[i].set_ylim([0, 1.05])

    # legend and xlabel
    for i in range(5, 9):
        ax[i].legend()
        ax[i].set_xlabel("conc. [nM]")
        ax[i].set_xticklabels(concsValues, rotation=30)



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
