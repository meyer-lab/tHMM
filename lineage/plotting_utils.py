''' Plotting utilities for lineages. Requires matplotlib, networkx, pygraphviz, and dot.'''

import networkx as nx
import math
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from networkx.drawing.nx_agraph import write_dot, graphviz_layout

def make_colormap_graph(X, X_like=None, prob=None, state=None, scale=300):
    '''
    Takes in a list of cells, and then outputs a color_map list and a list of cell indices.
    Example Usage:
    
    import matplotlib as mpl

    G, cmap, _ = make_colormap_graph(X)
    M = G.number_of_edges()
    edge_weights = [d for (u,v,d) in G.edges.data('weight')]
    #pos prog options: neato, dot, twopi, circo (don't use), fdp (don't use), nop (don't use), wc (don't use), acyclic (don't use), gvpr (don't use), gvcolor (don't use), ccomps (don't use), sccmap (don't use), tred (don't use), sfdp (don't use), unflatten (don't use)
    pos = graphviz_layout(G, prog='twopi', root=0)
    plt.figure(figsize=(40,31))
    plt.figaspect(1)
    node_size = 100
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=cmap, alpha=1)
    edges = nx.draw_networkx_edges(G, pos, node_size=node_size, edge_color=edge_weights, edge_cmap=plt.cm.inferno_r, width=2)

    ax = plt.gca()
    ax.set_axis_off()
    cb = plt.colorbar(edges)
    cb.set_label(label=r'Experiment Time [hrs]', labelpad=45)
    plt.title('Simulated Heterogeneous (by Breadth) Lineage')
    plt.rcParams.update({'font.size': 64})
    plt.show()
    
    # plotting the fit
    
    G, cmap, _ = make_colormap_graph(X, tHMMobj.states[0]) # note additional argument
    M = G.number_of_edges()
    edge_weights = [d for (u,v,d) in G.edges.data('weight')]
    #pos prog options: neato, dot, twopi, circo (don't use), fdp (don't use), nop (don't use), wc (don't use), acyclic (don't use), gvpr (don't use), gvcolor (don't use), ccomps (don't use), sccmap (don't use), tred (don't use), sfdp (don't use), unflatten (don't use)
    pos = graphviz_layout(G, prog='twopi', root=0)
    plt.figure(figsize=(40,31))
    plt.figaspect(1)
    node_size = 100
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=cmap, alpha=1)
    edges = nx.draw_networkx_edges(G, pos, node_size=node_size, edge_color=edge_weights, edge_cmap=plt.cm.inferno_r, width=2)

    ax = plt.gca()
    ax.set_axis_off()
    cb = plt.colorbar(edges)
    cb.set_label(label=r'Experiment Time [hrs]', labelpad=45)
    plt.title('Estimated Fit (Breadth)')
    plt.rcParams.update({'font.size': 64})
    plt.show()

    '''

    G = nx.Graph()
    edge_color_map = []
    node_color_map = []
    node_size_map = []

    for cell in X:
        edge_color_map.append(cell.endT)
        cell_idx = X.index(cell)
        G.add_node(cell_idx)
        parent_cell_idx = None

        if prob is None and state is None and X_like is None: #plot cell true state
            plotter = cell.true_state
        elif X_like is not None: #plot viterbi
            plotter = X_like[cell_idx]
        elif X_like is None and prob is not None and state is not None: #plot likelihoods
            plotter = state

        if plotter:
            node_color_map.append('red')
        else:
            node_color_map.append('green')

        if X_like is None and prob is not None and state is not None:
            node_size_map.append(prob[cell_idx]*scale)

        if cell_idx == 0:
            pass
        elif cell_idx > 0:
            parent_cell_idx = X.index(cell.parent)
            if math.isnan(cell.tau):
                G.add_edge(parent_cell_idx, cell_idx, weight=cell.startT)
            else:
                G.add_edge(parent_cell_idx, cell_idx, weight=cell.startT)

    return(G, node_color_map, node_size_map)

def plot_experiments(lin, filename):
    """ Creates lineage plots for all the experimental data. """
    state_ID = []
    for cell in lin:
        state_ID.append(cell.true_state) # append a 0 or 1 based on the cell's true state
    G, cmap, _ = make_colormap_graph(lin, state=state_ID)
    M = G.number_of_edges()
    edge_weights = [d for (u,v,d) in G.edges.data('weight')]

    pos = graphviz_layout(G, prog='twopi', root=0)
    plt.figure(figsize=(7,6))
    plt.figaspect(1)
    nodes = nx.draw_networkx_nodes(G, pos, node_size=50, node_color=cmap, alpha=0.65)
    edges = nx.draw_networkx_edges(G, pos, node_size=100, edge_color=edge_weights, edge_cmap=plt.cm.viridis_r, width=2)

    ax = plt.gca()
    ax.set_axis_off()
    cb = plt.colorbar(edges)
    cb.set_label(label=r'Experiment Time [hrs]')
    plt.title('Experimental Lineage')
    plt.rcParams.update({'font.size': 12})
    plt.savefig(filename)
    plt.show()
