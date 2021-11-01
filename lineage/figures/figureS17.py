""" This file plots the sankey figures for transitioned cells. """
import pickle
import numpy as np
import plotly.graph_objects as go


def counts_transit_time(tHMM, num_states):
    """ This functions finds cell counts transitions at t=0, t=T1, and t=T2. """
    T1 = 48
    T2 = 96
    T_counts = np.zeros((2, num_states, num_states))
    for lineage in tHMM.X:
        for i, cell in enumerate(lineage.output_lineage):

            if cell.isLeaf():

                if cell.parent is not None:
                    if (0 < cell.parent.time.startT <= T1) and (T1 < cell.parent.time.endT < T2):
                        T_counts[1, cell.parent.state, cell.state] += 1
                        T_counts[0, cell.get_root_cell().state, cell.parent.state] += 1

                    elif cell.parent.parent is not None:
                        if (0 < cell.parent.parent.time.startT <= T1) and (T1 < cell.parent.parent.time.endT < T2):
                            T_counts[1, cell.parent.parent.state, cell.state] += 1
                            T_counts[0, cell.get_root_cell().state, cell.parent.parent.state] += 1

                        elif cell.parent.parent.parent is not None:
                            if (0 < cell.parent.parent.parent.time.startT <= T1) and (T1 < cell.parent.parent.parent.time.endT < T2):
                                T_counts[1, cell.parent.parent.parent.state, cell.state] += 1
                                T_counts[0, cell.get_root_cell().state, cell.parent.parent.parent.state] += 1
    return T_counts

def plot_Sankey_time(drug_name, condition, indx_condition):
    """ Given the number of cells transitioned in each generation, plots the sankey figures. """
    pik = open(str(drug_name) +".pkl", "rb")
    tHMMobj_list = []
    for i in range(4):
        tHMMobj_list.append(pickle.load(pik))
    if drug_name == "lapatinibs":
        numST = 6
    elif drug_name == "gemcitabines":
        numST = 5

    counts = counts_transit_time(tHMMobj_list[indx_condition], numST)

    # let's just plot control for now
    node = dict(
        pad = 15,
        thickness = 20,
        line = dict(color = "black", width = 0.5),
        label = ["S00", "S01", "S02", "S03", "S04", "S05",
                 "S10", "S11", "S12", "S13", "S14", "S15",
                 "S20", "S21", "S22", "S23", "S24", "S25"],
        color = 3*['lightblue', 'orange', 'lightgreen', 'red', 'purple', 'olive']
    )
    link = dict(
        source = 6*[0] + 6*[1] + 6*[2] + 6*[3] + 6*[4] + 6*[5] +
                 6*[6] + 6*[7] + 6*[8] + 6*[9] + 6*[10] + 6*[11],
        target = 6 * [6, 7, 8, 9, 10, 11] + 6 * [12, 13, 14, 15, 16, 17],
        value = list(counts[0, :, :].flatten()) + list(counts[1, :, :].flatten())
    )
    fig = go.Figure(data=go.Sankey(node=node, link=link))
    fig.update_layout(title_text="The flow of states over time" + str(condition), font_size=10)
    return fig
