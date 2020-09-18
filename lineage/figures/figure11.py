""" This file depicts the distribution of phase lengths versus the states for each concentration. """
import numpy as np
import itertools
import seaborn as sns

from ..Analyze import run_Analyze_over
from ..data.Lineage_collections import Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM
from .figureCommon import getSetup, subplotLabel

concs = ["cntrl", "Lapt 25uM", "Lapt 50uM", "Lapt 250uM", "cntrl", "Gem 5uM", "Gem 10uM", "Gem 30uM"]
data = [Lapatinib_Control + Gemcitabine_Control, Lapt25uM, Lapt50uM, Lap250uM, Lapatinib_Control + Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM]

# Run fitting
output = run_Analyze_over(data, np.repeat([3, 4], 4))
lapt_tHMMobj_list = [oo[0] for oo in output[0: 4]]
lapt_states_list = [oo[1] for oo in output[0: 4]]
gemc_tHMMobj_list = [oo[0] for oo in output[4: 8]]
gemc_states_list = [oo[1] for oo in output[4: 8]]

def makeFigure():
    """ Makes figure 11. """

    ax, f = getSetup((13.2, 6.66), (2, 4))
    subplotLabel(ax)

    # lapatinib
    for idx, lapt_tHMMobj in enumerate(lapt_tHMMobj_list): # for each concentration data
        lpt_g1 = []
        lpt_g2 = []
        for lin_indx, lin in enumerate(lapt_tHMMobj.X): # for each lineage list
            for cell_indx, cell in enumerate(lin.output_lineage): # for each cell in the lineage
                lpt_g1.append(cell.obs[2])
                lpt_g2.append(cell.obs[3])

        LAP_state = list(itertools.chain(*lapt_states_list[idx])) + list(itertools.chain(*lapt_states_list[idx]))
        LAP_phaseLength = lpt_g1 + lpt_g2
        Lpt_phase = len(lpt_g1) * ["G1"] + len(lpt_g2) * ["G2"]

        # plot lapatinib
        sns.stripplot(x=LAP_state, y=LAP_phaseLength, hue=Lpt_phase, size=1, palette="Set2", linewidth=0.05, dodge=True, ax=ax[idx])

        ax[idx].set_title(concs[idx])
        ax[idx+4].set_title(concs[idx+4])
        ax[idx].set_ylabel("phase lengths")
        ax[idx+4].set_ylabel("phase lengths")
        ax[idx].set_xlabel("state")
        ax[idx+4].set_xlabel("state")

    # gemcitabine
    for idx, gemc_tHMMobj in enumerate(gemc_tHMMobj_list):
        gmc_g1 = []
        gmc_g2 = []
        for lin_indx, lin in enumerate(gemc_tHMMobj.X):
            for cell_indx, cell in enumerate(lin.output_lineage):
                gmc_g1.append(cell.obs[2])
                gmc_g2.append(cell.obs[3])

        GEM_state = list(itertools.chain(*gemc_states_list[idx])) + list(itertools.chain(*gemc_states_list[idx]))
        GEM_phaseLength = gmc_g1 + gmc_g2
        GEM_phase = len(gmc_g1) * ["G1"] + len(gmc_g2) * ["G2"]
        sns.stripplot(x=GEM_state, y=GEM_phaseLength, hue=GEM_phase, size=1, palette="Set2", linewidth=0.05, dodge=True, ax=ax[idx+4])

    return f
