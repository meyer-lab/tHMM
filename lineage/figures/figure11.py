""" This file depicts the distribution of phase lengths versus the states for each concentration of lapatinib. """
import numpy as np
import seaborn as sns
import networkx as nx
import pygraphviz
from string import ascii_lowercase
import pickle

from ..tHMM import tHMM
from .figureCommon import getSetup, subplotLabel

concs = ["control", "lapatinib 25 nM", "lapatinib 50 nM", "lapatinib 250 nM", "control", "5 nM", "10 nM", "30 nM"]
concsValues = ["control", "25 nM", "50 nM", "250 nM"]

pik1 = open("lapatinibs.pkl", "rb")
lapt_tHMMobj_list = []
for i in range(4):
    lapt_tHMMobj_list.append(pickle.load(pik1))

T_lap = lapt_tHMMobj_list[0].estimate.T
num_states = lapt_tHMMobj_list[0].num_states


def makeFigure():
    """ Makes figure 11. """

    ax, f = getSetup((16, 6.0), (2, 5))
    plot_all(ax, num_states, lapt_tHMMobj_list, "Laptinib", concsValues)
    return f


def plot_all(ax, num_states, lapt_tHMMobj_list, Dname, concsValues):
    ax[4].axis("off")
    ax[9].axis("off")
    ax[4].text(-0.2, 1.25, ascii_lowercase[8], transform=ax[4].transAxes, fontsize=16, fontweight="bold", va="top")

    # lapatinib
    lpt_avg = np.zeros((4, num_states, 2))  # the avg lifetime: num_conc x num_states x num_phases
    bern_lpt = np.zeros((4, num_states, 2))  # bernoulli
    # print parameters and estimated values
    print(Dname, "\n the \u03C0: ", lapt_tHMMobj_list[0].estimate.pi, "\n the transition matrix: ", lapt_tHMMobj_list[0].estimate.T)

    for idx, lapt_tHMMobj in enumerate(lapt_tHMMobj_list):  # for each concentration data
        for i in range(num_states):
            lpt_avg[idx, i, 0] = 1 / (lapt_tHMMobj.estimate.E[i].params[2] * lapt_tHMMobj.estimate.E[i].params[3])  # G1
            lpt_avg[idx, i, 1] = 1 / (lapt_tHMMobj.estimate.E[i].params[4] * lapt_tHMMobj.estimate.E[i].params[5])  # G2
            # bernoullis
            for j in range(2):
                bern_lpt[idx, i, j] = lapt_tHMMobj.estimate.E[i].params[j]

        LAP_state, LAP_phaseLength, Lpt_phase = twice(lapt_tHMMobj)

        # plot lapatinib
        sns.stripplot(x=LAP_state, y=LAP_phaseLength, hue=Lpt_phase, size=1.5, palette="Set2", dodge=True, ax=ax[idx])

        ax[idx].set_title(concs[idx])
        ax[idx].set_ylabel("phase lengths [hr]")
        ax[idx].set_xlabel("state")
        ax[idx].set_ylim([0.0, 150.0])

    plotting(ax, lpt_avg, bern_lpt, concs, concsValues)


def plotting(ax, lpt_avg, bern_lpt, concs, concsValues):
    """ helps to avoid duplicating code for plotting the gamma-related emission results and bernoulli. """
    for i in range(num_states):  # lapatinib that has 3 states
        ax[5].plot(concs[0: 4], lpt_avg[:, i, 0], label="state " + str(i + 1), alpha=0.7)
        ax[5].set_title("G1 phase")
        ax[6].plot(concs[0: 4], lpt_avg[:, i, 1], label="state " + str(i + 1), alpha=0.7)
        ax[6].set_title("G2 phase")
        ax[7].plot(concs[0: 4], bern_lpt[:, i, 0], label="state " + str(i + 1), alpha=0.7)
        ax[7].set_title("G1 phase")
        ax[8].plot(concs[0: 4], bern_lpt[:, i, 1], label="state " + str(i + 1), alpha=0.7)
        ax[8].set_title("G2 phase")

    # ylim and ylabel
    for i in range(5, 7):
        ax[i].set_ylabel("progression rate 1/[hr]")
        ax[i].set_ylim([0, 0.05])

    # ylim and ylabel
    for i in range(7, 9):
        ax[i].set_ylabel("division probability")
        ax[i].set_ylim([0, 1.05])

    # legend and xlabel
    for i in range(5, 9):
        ax[i].legend()
        ax[i].set_xlabel("concentration [nM]")
        ax[i].set_xticklabels(concsValues, rotation=30)

    subplotLabel(ax)


def twice(tHMMobj):
    """ For each tHMM object, connects the state and the emissions. """
    g1 = []
    g2 = []
    state = []
    for lin in tHMMobj.X:  # for each lineage list
        for cell in lin.output_lineage:  # for each cell in the lineage
            state.append((cell.state + 1))
            if cell.obs[4] == 1:
                g1.append(cell.obs[2])
            else:
                g1.append(np.nan)
            if cell.obs[5] == 1:
                g2.append(cell.obs[3])
            else:
                g2.append(np.nan)

    states = state + state # accounts for both phases
    phaseLength = g1 + g2
    phase = len(g1) * ["G1"] + len(g2) * ["G2"]
    return states, phaseLength, phase


def plot_networkx(num_states, T, drug_name):
    """ This plots the Transition matrix for each condition. """
    G = nx.MultiDiGraph()
    num_states = T.shape[0]

    # node labels
    labels = {}
    for i in range(num_states):
        labels[i] = "state " + str(i + 1)

    cs = ['lightblue', 'orange', 'lightgreen', 'red', 'purple']

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


plot_networkx(T_lap.shape[0], T_lap, 'lapatinib')
