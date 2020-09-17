""" This file depicts the distribution of phase lengths versus the states. """
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
import seaborn as sns

from ..Analyze import run_Analyze_over
from ..data.Lineage_collections import Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM
from .figureCommon import getSetup, subplotLabel

data = [Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM]

lpt_g1 = []
lpt_g2 = []
lpt12_g1 = [] # for fig. 12
lpt12_g2 = [] # for fig. 12

gmc_g1 = []
gmc_g2 = []
gmc12_g1 = [] # for fig. 12
gmc12_g2 = [] # for fig. 12

# Run fitting
output = run_Analyze_over(data, np.repeat([4, 3], 4))
gemc_tHMMobj_list = [output[i][0] for i in range(4)]
gemc_states_list = [output[i][1] for i in range(4)]
lapt_tHMMobj_list = [output[i][0] for i in range(4, 8)]
lapt_states_list = [output[i][1] for i in range(4, 8)]

for idx, lapt_tHMMobj in enumerate(lapt_tHMMobj_list):
    for lin_indx, lin in enumerate(lapt_tHMMobj.X):
        for cell_indx, cell in enumerate(lin.output_lineage):
            cell.state = lapt_states_list[idx][lin_indx][cell_indx]

            if np.isfinite(cell.obs[2]): # g1 length is not nan
                lpt_g1.append([cell.state, cell.obs[2]])
            if np.isfinite(cell.obs[3]): # g2 length is not nan
                lpt_g2.append([cell.state, cell.obs[3]])

    lpt12_g1.append(lpt_g1)
    lpt12_g2.append(lpt_g2)

for idx, gemc_tHMMobj in enumerate(gemc_tHMMobj_list):
    for lin_indx, lin in enumerate(gemc_tHMMobj.X):
        for cell_indx, cell in enumerate(lin.output_lineage):
            cell.state = gemc_states_list[idx][lin_indx][cell_indx]

            if np.isfinite(cell.obs[2]): # g1 length is not nan
                gmc_g1.append([cell.state, cell.obs[2]])
            if np.isfinite(cell.obs[3]): # g2 length is not nan
                gmc_g2.append([cell.state, cell.obs[3]])

    gmc12_g1.append(gmc_g1)
    gmc12_g2.append(gmc_g2)

def makeFigure():
    """
    Makes figure 11.
    """
    ax, f = getSetup((7, 3), (1, 2))
    subplotLabel(ax)

    Lap_data = pd.DataFrame(columns=["state", "phase lengths [hrs]", "phase"])
    Lap_data["state"] = [a[0] for a in lpt_g1] + [a[0] for a in lpt_g2]
    Lap_data["phase lengths [hrs]"] = [a[1] for a in lpt_g1] + [a[1] for a in lpt_g2]
    Lap_data["phase"] = len(lpt_g1) * ["G1"] + len(lpt_g2) * ["G2"]

    Gem_data = pd.DataFrame(columns=["state", "phase lengths [hrs]", "phase"])
    Gem_data["state"] = [a[0] for a in gmc_g1] + [a[0] for a in gmc_g2]
    Gem_data["phase lengths [hrs]"] = [a[1] for a in gmc_g1] + [a[1] for a in gmc_g2]
    Gem_data["phase"] = len(gmc_g1) * ["G1"] + len(gmc_g2) * ["G2"]

    # plot
    sns.stripplot(x="state", y="phase lengths [hrs]", hue="phase", data=Lap_data, size=1, palette="Set2", linewidth=0.05, dodge=True, ax=ax[0])
    ax[0].set_title("Lapatinib")
    ax[1].set_title("Gemcitabine")
    
    sns.stripplot(x="state", y="phase lengths [hrs]", hue="phase", data=Gem_data, palette="Set2", size=1, linewidth=0.05, dodge=True, ax=ax[1])

    return f
