""" This file plots the sankey figures for MCF10A transitioned cells. """
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from .common import getSetup

pik1 = open("gf.pkl", "rb")
gf_tHMMobj_list = []
for i in range(4):
    gf_tHMMobj_list.append(pickle.load(pik1))

def makeFigure():
    """
    Makes figure 90.
    """
    num_states = 6
    ax, f = getSetup((10, 4), (1, 2))
    migration(ax, num_states, gf_tHMMobj_list)
    return f

def migration(ax, num_states, tHMMobj_list):
    vel_avg = np.zeros((4, num_states))
    dist_avg = np.zeros((4, num_states))

    for idx, tHMMobj in enumerate(tHMMobj_list):
        sts = []
        vels = []
        dists = []
        for lineage in tHMMobj.X:
            for cell in lineage.output_lineage:
                sts.append(cell.state)
                vels.append(cell.obs[3])
                dists.append(cell.obs[4])
                assert not np.isnan(cell.obs[3])
                assert not np.isnan(cell.obs[4])
            df_temp = pd.DataFrame({'states': sts, 'velocity':vels, 'distance':dists})
            for i in range(num_states):
                vel_avg[idx, i] = np.mean(df_temp.loc[df_temp['states']==i]['velocity'])
                dist_avg[idx, i] = np.mean(df_temp.loc[df_temp['states']==i]['distance'])

    df1 = pd.DataFrame({'Growth Factors': ['PBS', 'EGF', 'HGF', 'OSM'],
                       'State1': vel_avg[:, 0], 
                       'State2': vel_avg[:, 1],
                       'State3': vel_avg[:, 2],
                       'State4': vel_avg[:, 3],
                       'State5': vel_avg[:, 4],
                       'State6': vel_avg[:, 5]})

    df2 = pd.DataFrame({'Growth Factors': ['PBS', 'EGF', 'HGF', 'OSM'],
                       'State1': dist_avg[:, 0], 
                       'State2': dist_avg[:, 1],
                       'State3': dist_avg[:, 2],
                       'State4': dist_avg[:, 3],
                       'State5': dist_avg[:, 4],
                       'State6': dist_avg[:, 5]})

    df1[['Growth Factors', 'State1', 'State2', 'State3', 'State4', 'State5', 'State6']].plot(x='Growth Factors', kind='bar', rot=0, ax=ax[0])
    df2[['Growth Factors', 'State1', 'State2', 'State3', 'State4', 'State5', 'State6']].plot(x='Growth Factors', kind='bar', rot=0, ax=ax[1])
    ax[0].set_title("Avg Velocity")
    ax[0].set_ylabel("cm/hr?")
    ax[1].set_title("Avg Distance")
    ax[1].set_ylabel("cm?")

def counts_transit_time(tHMM, num_states):
    """ This functions finds cell counts transitions at t=0, t=T1, and t=T2. """
    T1 = 24
    T2 = 48
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
    pik = open(str(drug_name) + ".pkl", "rb")
    tHMMobj_list = []
    for i in range(4):
        tHMMobj_list.append(pickle.load(pik))

    if indx_condition == 0:

        label = ["S00", "S01", "S02", "S03", "S04", "S05", "S06",
                    "S10", "S11", "S12", "S13", "S14", "S15", "S16"]
        numST = 7
        source = []
        for i in range(6):
            source += numST * [i]

        target = numST * [7, 8, 9, 10, 11, 12, 13]
        color = 2 * ['lightblue', 'orange', 'lightgreen', 'red', 'purple', 'olive', 'pink']

    else:
        label = ["S00", "S01", "S02", "S03", "S04", "S05", "S06",
                    "S10", "S11", "S12", "S13", "S14", "S15", "S16",
                    "S20", "S21", "S22", "S23", "S24", "S25", "S26"]
        numST = 7
        source = []
        for i in range(13):
            source += numST * [i]

        target = numST * [7, 8, 9, 10, 11, 12, 13] + numST * [14, 15, 16, 17, 18, 19, 20]
        color = 3 * ['lightblue', 'orange', 'lightgreen', 'red', 'purple', 'olive', 'pink']
    counts = counts_transit_time(tHMMobj_list[indx_condition], numST)

    node = dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=label,
        color=color
    )
    link = dict(
        source=source,
        target=target,
        value=list(counts[0, :, :].flatten()) + list(counts[1, :, :].flatten())
    )
    fig = go.Figure(data=go.Sankey(node=node, link=link))
    fig.update_layout(title_text="The flow of states over time" + str(condition), font_size=10)
    return fig
