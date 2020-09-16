""" This file depicts the distribution of phase lengths versus the states. """
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
import seaborn as sns

from ..Analyze import run_Analyze_over
from ..data.Lineage_collections import Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM
from .figureCommon import getSetup, subplotLabel

data = [Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM]

lapatinib = []
lapatinib12 = []
gemcitabine = []
gemcitabine12 = []
# Run fitting
output = run_Analyze_over(data, np.repeat([4, 3], 4))
gemc_tHMMobj_list = [output[i][0] for i in range(4)]
gemc_states_list = [output[i][1] for i in range(4)]
lapt_tHMMobj_list = [output[i][0] for i in range(4, 8)]
lapt_states_list = [output[i][1] for i in range(4, 8)]

for idx, lapt_tHMMobj in enumerate(lapt_tHMMobj_list):
    for lin_indx, lin in enumerate(lapt_tHMMobj.X):
        level1 = []
        for cell_indx, cell in enumerate(lin.output_lineage):
            cell.state = lapt_states_list[idx][lin_indx][cell_indx]
            lapatinib.append([cell.state, cell.obs[2], cell.obs[3]])
    lapatinib12.append(lapatinib)

for idx, gemc_tHMMobj in enumerate(gemc_tHMMobj_list):
    for lin_indx, lin in enumerate(gemc_tHMMobj.X):
        for cell_indx, cell in enumerate(lin.output_lineage):
            cell.state = gemc_states_list[idx][lin_indx][cell_indx]
            gemcitabine.append([cell.state, cell.obs[2], cell.obs[3]])
    gemcitabine12.append(gemcitabine)

def makeFigure():
    """
    Makes figure 11.
    """
    ax, f = getSetup((7, 6), (2, 2))
    subplotLabel(ax)

    Lap_data = pd.DataFrame(columns=["state", "G1 length", "G2 length"])
    Lap_data["state"] = [a[0] for a in lapatinib]
    Lap_data["G1 lengths"] = [a[1] for a in lapatinib]
    Lap_data["G2 lengths"] = [a[2] for a in lapatinib]

    Gem_data = pd.DataFrame(columns=["state", "G1 length", "G2 length"])
    Gem_data["state"] = [a[0] for a in gemcitabine]
    Gem_data["G1 lengths"] = [a[1] for a in gemcitabine]
    Gem_data["G2 lengths"] = [a[2] for a in gemcitabine]

    # plot
    sns.stripplot(x="state", y="G1 lengths", data=Lap_data, linewidth=1, ax=ax[0], jitter=0.1)
    sns.stripplot(x="state", y="G2 lengths", data=Lap_data, linewidth=1, ax=ax[1], jitter=0.1)
    ax[0].set_ylabel("G1 phase lengths")
    ax[1].set_ylabel("G2 phase lengths")
    for i in range(4):
        ax[i].set_ylim(bottom=-20, top=180)
    for j in range(2):
        ax[j].set_title("Lapatinib treatment")
        ax[j+2].set_title("Gemcitabine treatment")
    
    sns.stripplot(x="state", y="G1 lengths", data=Gem_data, linewidth=1, ax=ax[2], jitter=0.1)
    sns.stripplot(x="state", y="G2 lengths", data=Gem_data, linewidth=1, ax=ax[3], jitter=0.1)
    ax[2].set_ylabel("G1 phase lengths")
    ax[3].set_ylabel("G2 phase lengths")

    return f
