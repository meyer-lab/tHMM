''' Plotting utilities for lineages. Requires matplotlib, networkx, pygraphviz, and dot.'''


def lineage2Graph(X):
    '''
        Plots a lineage of cells.

        Example usage: (X is a lineage)

        G, cmap = lineage2Graph(X)
        pos=graphviz_layout(G, prog='dot')
        plt.figure(figsize=(31,10))
        nx.draw(G, pos, node_color = cmap)
    '''
    G = nx.Graph()
    color_map = []
    for cell in X:
        if cell.true_state:
            color_map.append('red')
        else:
            color_map.append('green')
        cell_idx = X.index(cell)
        G.add_node(cell_idx)
        parent_cell_idx = None
        if cell_idx == 0:
            pass
        else:
            parent_cell_idx = X.index(cell.parent)
            G.add_edge(parent_cell_idx, cell_idx)
    return(G, color_map)

def lineageLike2Graph(X, X_like):
    '''
        Plots an object (list) that has the same shape as the original lineage of cells
        where the values represent states (usually all_states[i]).

        Example usage: (X is a lineage, X_like is all_states[0])

        G, cmap = lineageLike2Graph(X,all_states[0])
        pos=graphviz_layout(G, prog='dot')
        plt.figure(figsize=(31,10))
        nx.draw(G, pos, node_color = cmap)
    '''
    G = nx.Graph()
    color_map = []
    for cell in X:
        cell_idx = X.index(cell)
        if X_like[cell_idx]:
            color_map.append('red')
        else:
            color_map.append('green')
        G.add_node(cell_idx)
        parent_cell_idx = None
        if cell_idx == 0:
            pass
        else:
            parent_cell_idx = X.index(cell.parent)
            G.add_edge(parent_cell_idx, cell_idx)
    return(G, color_map)

def lineageProb2Graph(X, prob, state, scale=300):
    '''
        Plots an object (list) that has the same shape as the original lineage of cells
        where the values are probabilities between 0 and 1. Usually, this can come from any
        of the several values calculated in the upward or downward recursions like betas, gammas, or
        stored in the tHMM class like MSD, EL, or used in Viterbi, like deltas.

        Example usage: (X is a lineage, tHMMobj.EL[0][:,0] for the prob argument are likelihoods, state 0 is being used
        and scale is to appropriately size the nodes)

        G, cmap, node_size_map = lineageProb2Graph(X,tHMMobj.EL[0][:,0],0, scale=150)
        pos=graphviz_layout(G, prog='dot')
        plt.figure(figsize=(31,10))
        nx.draw(G, pos, node_color = cmap, node_size=node_size_map)

    '''
    G = nx.Graph()
    color_map = []
    node_size_map = []
    for cell in X:
        cell_idx = X.index(cell)
        if state:
            color_map.append('red')
        else:
            color_map.append('green')
        node_size_map.append(prob[cell_idx]*scale)
        G.add_node(cell_idx)
        parent_cell_idx = None
        if cell_idx == 0:
            pass
        else:
            parent_cell_idx = X.index(cell.parent)
            G.add_edge(parent_cell_idx, cell_idx)
    return(G, color_map, node_size_map)
