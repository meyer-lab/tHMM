""" To draw transition matrix """
import numpy as np
import seaborn as sns
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph

from .figure11 import gemc_tHMMobj_list, lapt_tHMMobj_list
from .figureCommon import getSetup, subplotLabel

gemc = gemc_tHMMobj_list[0]
lapt = lapt_tHMMobj_list[0]
T_lap = lapt_tHMMobj_list[0].estimate.T
T_gem = gemc_tHMMobj_list[0].estimate.T

def makeFigure():
    """ makes figure 13 for transition matrices. """

    ax, f = getSetup((13, 8), (1, 2))
    subplotLabel(ax)

    # transition matrix lapatinib
    plot_networkx(T_lap.shape[0], T_lap, "lpt")
    ax[0].axis("off")
    ax[0].set_title("lapatinib")

    # transition matrix
    plot_networkx(T_gem.shape[0], T_gem, "gmc")
    ax[1].axis("off")
    ax[1].set_title("gemcitabine")

    return f

def plot_networkx(num_states, T, drug_name):
    """ This plots the Transition matrix for each condition. """
    G = nx.MultiDiGraph()
    num_states = T.shape[0]

    # node labels
    labels = {}
    for i in range(num_states):
        labels[i] = "state " + str(i + 1)

    # add nodes
    for i in range(num_states):
        G.add_node(i, pos=(-2, -2), label=labels[i],style='filled',fillcolor='lightblue')

    # add edges
    for i in range(num_states):
        for j in range(num_states):
            G.add_edge(i, j, penwidth=2*[i, j], minlen=1)

    # add graphviz layout options (see https://stackoverflow.com/a/39662097)
    G.graph['edge'] = {'arrowsize': '0.6', 'splines': 'curved'}
    G.graph['graph'] = {'scale': '1'}

    # adding attributes to edges in multigraphs is more complicated but see
    # https://stackoverflow.com/a/26694158                    
    for i in range(num_states):
        G[i][i][0]['color']='black'

    A = to_agraph(G) 
    A.layout('dot')                                                                 
    A.draw('output/'+str(drug_name)+'.svg')