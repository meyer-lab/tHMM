""" This file is only to plot the distribution of death lengths. """
import numpy as np
import seaborn as sns
import networkx as nx

from .figure11 import gemc_tHMMobj_list, lapt_tHMMobj_list
from .figureCommon import getSetup, subplotLabel

gemc = gemc_tHMMobj_list[-1]
lapt = lapt_tHMMobj_list[-2]

gm = []
lp = []
for lin in gemc.X:
    for cell in lin.output_lineage:
        if cell.obs[0] == 0 or cell.obs[1] == 0:
            if np.isfinite(cell.obs[2]) and np.isfinite(cell.obs[3]):
                length = cell.obs[2] + cell.obs[3]
            elif np.isnan(cell.obs[2]):
                length = cell.obs[3]
            elif np.isnan(cell.obs[3]):
                length = cell.obs[2]
            gm.append(length)

for lin in lapt.X:
    for cell in lin.output_lineage:
        if cell.obs[0] == 0 or cell.obs[1] == 0:
            if np.isfinite(cell.obs[2]) and np.isfinite(cell.obs[3]):
                length = cell.obs[2] + cell.obs[3]
            elif np.isnan(cell.obs[2]):
                length = cell.obs[3]
            elif np.isnan(cell.obs[3]):
                length = cell.obs[2]
            lp.append(length)

print("gem", len(gm))
print("lap", len(lp))

def makeFigure():
    """ Makes figure 11. """

    ax, f = getSetup((13.2, 4.5), (1, 4))
    subplotLabel(ax)
    ax[0].hist(gm, bins=np.max(gm)-np.min(gm))
    ax[1].hist(lp, bins=np.max(lp)-np.min(lp))
    ax[1].set_title("lapatinib")
    ax[0].set_title("gemcitabine")

    for i in range(2):
        ax[i].set_ylabel("freq.")
        ax[i].set_xlabel("lived before death")
    # get the transition matrix
    T_lap = lapt_tHMMobj_list[0].estimate.T
    T_gem = gemc_tHMMobj_list[0].estimate.T

    # transition matrix lapatinib
    plot_networkx(T_lap.shape[0], 8*T_lap, ax[2])
    ax[2].set_title("lapatinib")


    # transition matrix
    plot_networkx(T_gem.shape[0], 8*T_gem, ax[3])
    ax[3].set_title("gemcitabine")


    return f

def plot_networkx(num_states, T, axes):
    """ This plots the Transition matrix for each condition. """
    G=nx.DiGraph()

    # add nodes
    for i in range(num_states):
        G.add_node(i, pos=(i,i))

    # add edges
    for i in range(num_states):
        for j in range(num_states):
            G.add_edge(i, j, weight=T[i,j], length=5)

    # set circular position
    pos=nx.circular_layout(G)

    # node labels
    labels={}
    for i in range(num_states):
        labels[i] = "state "+str(i+1)

    # set line thickness as the transition values
    weights = nx.get_edge_attributes(G,'weight').values()
    nx.draw(G, pos, arrowsize=20, font_size=9, font_color="w", labels=labels, node_size=600, alpha=0.9,width=list(weights), with_labels=True,connectionstyle='arc3, rad = 0.2', ax=axes)

    