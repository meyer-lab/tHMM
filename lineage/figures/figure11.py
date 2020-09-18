""" This file depicts the distribution of phase lengths versus the states for each concentration. """
import numpy as np
import pandas as pd
import itertools
from matplotlib.ticker import MaxNLocator
import seaborn as sns

from ..Analyze import run_Analyze_over
from ..data.Lineage_collections import Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM
from .figureCommon import getSetup, subplotLabel

concs = ["cntrl", "Lapt 25uM", "Lapt 50uM", "Lapt 250uM", "cntrl", "Gem 5uM", "Gem 10uM", "Gem 30uM"]
data = [Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM, Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM]

# Run fitting
output = run_Analyze_over(data, np.repeat([3, 4], 4))
lapt_tHMMobj_list = [oo[0] for oo in output[0: 4]]
lapt_states_list = [oo[1] for oo in output[0: 4]]
gemc_tHMMobj_list = [oo[0] for oo in output[4: 8]]
gemc_states_list = [oo[1] for oo in output[4: 8]]

# lapatinib
LAP = pd.DataFrame(columns=["state", "phase"])
for idx, lapt_tHMMobj in enumerate(lapt_tHMMobj_list): # for each concentration data
    lpt_g1 = []
    lpt_g2 = []
    for lin_indx, lin in enumerate(lapt_tHMMobj.X): # for each lineage list
        for cell_indx, cell in enumerate(lin.output_lineage): # for each cell in the lineage
            lpt_g1.append(cell.obs[2])
            lpt_g2.append(cell.obs[3])

    LAP["state "+str(concs[idx])] = list(itertools.chain(*lapt_states_list[idx]))
    LAP["phase lengths "+str(concs[idx])] = lpt_g1 + lpt_g2
    LAP["phase "+str(concs[idx])] = len(lpt_g1) * ["G1"] + len(lpt_g2) * ["G2"]

# gemcitabine
GEM = pd.DataFrame(columns=["state", "phase"])   
for idx, gemc_tHMMobj in enumerate(gemc_tHMMobj_list):
    gmc_g1 = []
    gmc_g2 = []
    for lin_indx, lin in enumerate(gemc_tHMMobj.X):
        for cell_indx, cell in enumerate(lin.output_lineage):
            gmc_g1.append(cell.obs[2])
            gmc_g2.append(cell.obs[3])

    GEM["state "+str(concs[idx+4])] = list(itertools.chain(*gemc_states_list[idx]))
    GEM["phase lengths "+str(concs[idx+4])] = gmc_g1 + gmc_g2
    GEM["phase "+str(concs[idx+4])] = len(gmc_g1) * ["G1"] + len(gmc_g2) * ["G2"]

def makeFigure():
    """
    Makes figure 11.
    """
    ax, f = getSetup((13.2, 6.66), (2, 4))
    subplotLabel(ax)

    for i in range(4):
        sns.stripplot(x="state "+str(concs[i]), y="phase lengths "+str(concs[i]), hue="phase "+str(concs[i]), data=LAP, size=1, palette="Set2", linewidth=0.05, dodge=True, ax=ax[i])
        sns.stripplot(x="state "+str(concs[i+4]), y="phase lengths "+str(concs[i+4]), hue="phase "+str(concs[i+4]), data=GEM, size=1, palette="Set2", linewidth=0.05, dodge=True, ax=ax[i+4])

        ax[i].set_title(concs[i])
        ax[i+4].set_title(concs[i+4])
        ax[i].set_ylabel("phase lengths")
        ax[i+4].set_ylabel("phase lengths")

        # this removes title of legends
        handles, labels = ax[i].get_legend_handles_labels()
        ax[i].legend(handles=handles[1:], labels=labels[1:])
        handles, labels = ax[i+4].get_legend_handles_labels()
        ax[i+4].legend(handles=handles[1:], labels=labels[1:])

    return f
