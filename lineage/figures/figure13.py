""" To draw transition matrix """
import numpy as np
import seaborn as sns
import networkx as nx

from .figure11 import gemc_tHMMobj_list, lapt_tHMMobj_list
from .figureCommon import getSetup, subplotLabel

gemc = gemc_tHMMobj_list[0]
lapt = lapt_tHMMobj_list[0]

def makeFigure():
    """ makes figure 13 for transition matrices. """

    ax, f = getSetup((16, 20), (2, 1))
    # subplotLabel(ax)
    T_lap = lapt_tHMMobj_list[0].estimate.T
    T_gem = gemc_tHMMobj_list[0].estimate.T

    # transition matrix lapatinib
    plot_networkx(T_lap.shape[0], 8*T_lap, ax[0])
    ax[0].set_title("lapatinib")


    # transition matrix
    plot_networkx(T_gem.shape[0], 8*T_gem, ax[1])
    ax[1].set_title("gemcitabine")


    return f

def plot_networkx(num_states, T, axes):
    """ This plots the Transition matrix for each condition. """
    G=nx.DiGraph()

    # add nodes
    for i in range(num_states):
        G.add_node(i, pos=(-2,-2))

    # add edges
    for i in range(num_states):
        for j in range(num_states):
            G.add_edge(i, j, weight=T[i,j], minlen=1)

    # set circular position
    pos=nx.circular_layout(G)

    # node labels
    labels={}
    for i in range(num_states):
        labels[i] = "state "+str(i+1)

    # set line thickness as the transition values
    weights = nx.get_edge_attributes(G,'weight').values()
    nx.draw(G, pos, arrowsize=20, font_size=14, font_color="w", font_weight="bold", labels=labels, node_size=4700, alpha=0.9,width=list(weights), with_labels=True,connectionstyle='arc3, rad = 0.2', ax=axes)
