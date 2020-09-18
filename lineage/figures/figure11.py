""" This file depicts the distribution of phase lengths versus the states for each concentration. """
import numpy as np
import itertools
import seaborn as sns

from ..Analyze import run_Analyze_over
from ..data.Lineage_collections import Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM
from .figureCommon import getSetup, subplotLabel

concs = ["cntrl", "Lapt 25nM", "Lapt 50nM", "Lapt 250nM", "cntrl", "Gem 5nM", "Gem 10nM", "Gem 30nM"]
data = [Lapatinib_Control + Gemcitabine_Control, Lapt25uM, Lapt50uM, Lap250uM, Gemcitabine_Control + Lapatinib_Control, Gem5uM, Gem10uM, Gem30uM]

# Run fitting
output = run_Analyze_over(data, np.repeat([3, 4], 4))
lapt_tHMMobj_list = [oo[0] for oo in output[0: 4]]
lapt_states_list = [oo[1] for oo in output[0: 4]]
gemc_tHMMobj_list = [oo[0] for oo in output[4: 8]]
gemc_states_list = [oo[1] for oo in output[4: 8]]

def twice(tHMMobj, state):
    g1 = []
    g2 = []
    for lin_indx, lin in enumerate(tHMMobj.X): # for each lineage list
        for cell_indx, cell in enumerate(lin.output_lineage): # for each cell in the lineage
            g1.append(cell.obs[2])
            g2.append(cell.obs[3])

    state = list(itertools.chain(*state)) + list(itertools.chain(*state))
    phaseLength = g1 + g2
    phase = len(g1) * ["G1"] + len(g2) * ["G2"]
    return state, phaseLength, phase


def makeFigure():
    """ Makes figure 11. """

    ax, f = getSetup((13.2, 6.66), (2, 4))
    subplotLabel(ax)

    # lapatinib
    print("lapatinib, 3 states: \n")
    for idx, lapt_tHMMobj in enumerate(lapt_tHMMobj_list): # for each concentration data
        # print parameters and estimated values
        print("for concentration ", concs[idx], "\n the \u03C0: ", lapt_tHMMobj.estimate.pi, "\n the transition matrix: ", lapt_tHMMobj.estimate.T)
        for i in range(3):
            print("\n parameters for state ", i, " are: ", lapt_tHMMobj.estimate.E[i].params)
        LAP_state, LAP_phaseLength, Lpt_phase = twice(lapt_tHMMobj, lapt_states_list[idx])

        # plot lapatinib
        sns.stripplot(x=LAP_state, y=LAP_phaseLength, hue=Lpt_phase, size=1, palette="Set2", linewidth=0.05, dodge=True, ax=ax[idx])

        ax[idx].set_title(concs[idx])
        ax[idx+4].set_title(concs[idx+4])
        ax[idx].set_ylabel("phase lengths")
        ax[idx+4].set_ylabel("phase lengths")
        ax[idx].set_xlabel("state")
        ax[idx+4].set_xlabel("state")
        ax[idx].set_ylim([0, 160])
        ax[idx+4].set_ylim([0, 160])

    # gemcitabine
    print("Gemcitabine, 4 states: \n")
    for idx, gemc_tHMMobj in enumerate(gemc_tHMMobj_list):
        # print parameters and estimated values
        print("for concentration ", concs[idx+4], "\n the \u03C0: ", gemc_tHMMobj.estimate.pi, " \n the transition matrix: ", gemc_tHMMobj.estimate.T)
        for i in range(4):
            print("\n parameters for state ", i, " are: ", gemc_tHMMobj.estimate.E[i].params)
        GEM_state, GEM_phaseLength, GEM_phase = twice(gemc_tHMMobj, gemc_states_list[idx])
        sns.stripplot(x=GEM_state, y=GEM_phaseLength, hue=GEM_phase, size=1, palette="Set2", linewidth=0.05, dodge=True, ax=ax[idx+4])

    return f
