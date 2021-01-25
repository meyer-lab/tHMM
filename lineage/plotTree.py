""" In this file we plot! """

import numpy as np
from Bio.Phylo.BaseTree import Clade
from Bio import Phylo
from matplotlib import pylab
import networkx as nx
import pygraphviz


def CladeRecursive(cell, a, censore, color):
    """ To plot the lineage while censored (from G1 or G2).
    If cell died in G1, the lifetime of the cell until dies is shown in red.
    If cell died in G2, the lifetime of the cell until dies is shown in blue.
    If none of the above, the cell continues to divide and is shown in black.
    a should be: a = [Clade(lineage1.full_lineage[0].obs[2]+lineage1.full_lineage[0].obs[3])]
    If you are interested, you can take a look at the source code for creating Clades manually:
    https://github.com/biopython/biopython/blob/fce4b11b4b8e414f1bf093a76e04a3260d782905/Bio/Phylo/BaseTree.py#L801
    """
    if color:
        if cell.state == 0:
            colorr = "blue"
        elif cell.state == 1:
            colorr = "green"
        elif cell.state == 2:
            colorr = "red"
        elif cell.state == 3:
            colorr = "yellow"
        elif cell.state == 4:
            colorr = "black"
        elif cell.state == 5:
            colorr = "cyan"
        elif cell.state == 6:
            colorr = "pink"
        else:
            colorr = "brown"
    else:
        colorr = "black"

    if cell.isLeaf() and censore:
        if np.isfinite(cell.obs[2]) and np.isfinite(cell.obs[3]):
            length = cell.obs[2] + cell.obs[3]
        elif np.isnan(cell.obs[2]):
            length = cell.obs[3]
        elif np.isnan(cell.obs[3]):
            length = cell.obs[2]
        return Clade(branch_length=length, width=1, color=colorr)

    else:
        clades = []
        if cell.left is not None and cell.left.observed:
            clades.append(CladeRecursive(cell.left, a, censore, color))
        if cell.right is not None and cell.right.observed:
            clades.append(CladeRecursive(cell.right, a, censore, color))
        if np.isnan(cell.obs[3]):  # if the cell got stuck in G1
            lengths = cell.obs[2]
        elif np.isnan(cell.obs[2]):  # is a root parent and G1 is not observed
            lengths = cell.obs[3]
        else:
            lengths = cell.obs[2] + cell.obs[3]  # both are observed
        return Clade(branch_length=lengths, width=1, clades=clades, color=colorr)


def plotLineage(lineage, axes, censore=True, color=True):
    """
    Makes lineage tree.
    """

    root = lineage.output_lineage[0]
    if np.isfinite(root.obs[4]):  # starts from G1
        if np.isfinite(root.obs[3]):
            length = root.obs[2] + root.obs[3]
        else:
            length = root.obs[2]
        assert np.isfinite(length)
    else:  # starts from G2
        length = root.obs[3]
        assert np.isfinite(length)
    a = [Clade(length)]

    # input the root cells in the lineage
    c = CladeRecursive(lineage.output_lineage[0], a, censore, color)

    return Phylo.draw(c, axes=axes)


def plot_networkx(num_states, T, drug_name):
    """ This plots the Transition matrix for each condition. """
    G = nx.MultiDiGraph()
    num_states = T.shape[0]

    # node labels
    labels = {}
    for i in range(num_states):
        labels[i] = "state " + str(i + 1)

    cs = ['lightblue', 'orange', 'lightgreen', 'red', 'purple', 'olive', 'gray']

    # add nodes
    for i in range(num_states):
        G.add_node(i, pos=(-2, -2), label=labels[i], style='filled', fillcolor=cs[i])

    # add edges
    for i in range(num_states):
        for j in range(num_states):
            G.add_edge(i, j, penwidth=2 * T[i, j], minlen=1, label=str(np.round(T[i, j], 2)))

    # add graphviz layout options (see https://stackoverflow.com/a/39662097)
    G.graph['edge'] = {'arrowsize': '0.6', 'splines': 'curved'}
    G.graph['graph'] = {'scale': '1'}

    # adding attributes to edges in multigraphs is more complicated but see
    # https://stackoverflow.com/a/26694158
    for i in range(num_states):
        G[i][i][0]['color'] = 'black'

    A = nx.drawing.nx_agraph.to_agraph(G)
    A.layout('dot')
    A.draw('lineage/figures/cartoons/' + str(drug_name) + '.svg')
