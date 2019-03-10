''' Plotting utilities for lineages. Requires matplotlib, networkx, pygraphviz, and dot.'''

import networkx as nx

def make_colormap_graph(X, X_like, prob, state, scale=300):
    '''Takes in a list of cells, and then outputs a color_map list and a list of cell indices'''

    G = nx.Graph()
    node_color_map = []
    node_size_map = []

    for cell in X:
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
            print(prob[cell_idx])
            node_size_map.append(prob[cell_idx]*scale)

        if cell_idx == 0:
            pass
        else:
            parent_cell_idx = X.index(cell.parent)
            G.add_edge(parent_cell_idx, cell_idx)

    return(G, color_map, node_size_map)
