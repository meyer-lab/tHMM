""" This file contains figures related to how far the states need to be,
which is shown by Wasserestein distance. """
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score
import itertools

from .figureCommon import (
    getSetup,
    subplotLabel,
    commonAnalyze,
    pi,
    T,
    max_desired_num_cells,
    num_data_points,
    scatter_kws_list,
)
from ..tHMM import tHMM
from ..LineageTree import LineageTree
from ..states.StateDistributionGaPhs import StateDistribution


def makeFigure():
    """
    Makes fig 6.
    """

    # Get list of axis objects
    ax, f = getSetup((7, 5), (2, 2))
    figureMaker5(ax, *accuracy())

    subplotLabel(ax)

    return f


def accuracy():
    """ A Helper function to create more random copies of a population. """
    # Creating a list of populations to analyze over
    list_of_Es = [[StateDistribution(0.99, 0.9, 12, a, 8, 5), StateDistribution(0.99, 0.8, 12, 1.5, 8, 5)] for a in np.linspace(1.5, 4, num_data_points)]
    list_of_fpi = [pi] * len(list_of_Es)
    list_of_populations = [[LineageTree.init_from_parameters(pi, T, E, max_desired_num_cells)] for E in list_of_Es]

    allpops=[]
    balanced_score = []
    true = []
    all_cells=[]

    for pop in list_of_populations:
        thmmobj = tHMM(pop, num_states=2)
        true_states_by_lineage = [[cell.state for cell in lineage.output_lineage] for lineage in thmmobj.X]
        tmp = [state for sublist in true_states_by_lineage for state in sublist]
        true.append(tmp)
        ravel_true_states = np.array(list(itertools.chain.from_iterable(true)))

        tmp = [cell.obs for lineage in thmmobj.X for cell in lineage.output_lineage]
        all_cells.append(tmp)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(np.array(list(itertools.chain.from_iterable(all_cells)))).labels_
        balanced_score.append(100 * balanced_accuracy_score(ravel_true_states, kmeans))


    wass, _, accuracy, _, _, _ = commonAnalyze(list_of_populations, 2, xtype="wass", list_of_fpi=list_of_fpi, parallel=True)

    distribution_df = pd.DataFrame(columns=["Distribution type", "G1 lifetime", "State"])
    lineages = [list_of_populations[int(num_data_points * i / 4.)][0] for i in range(4)]
    len_lineages = [len(lineage) for lineage in lineages]
    distribution_df["G1 lifetime"] = [(cell.obs[1] + cell.obs[2]) for lineage in lineages for cell in lineage.output_lineage]
    distribution_df["State"] = ["State 1" if cell.state == 0 else "State 2" for lineage in lineages for cell in lineage.output_lineage]
    distribution_df["Distribution type"] = len_lineages[0] * ["Same\n"+str(0)+"-"+str(wass[-1]/4)] +\
        len_lineages[1] * ["Similar\n"+str(wass[-1]/4)+"-"+str(wass[-1]/2)] +\
        len_lineages[2] * ["Different\n"+str(wass[-1]/2)+"-"+str(wass[-1]*0.75)] +\
        len_lineages[3] * ["Distinct\n>"+str(wass[-1]*0.75)]

    # for the violin plot (distributions)
    wasser_df = pd.DataFrame(columns=["Wasserstein distance", "State Assignment Accuracy"])
    wasser_df["Wasserstein distance"] = wass
    wasser_df["State Assignment Accuracy"] = accuracy
    wasser_df["KMeans accuracy"] = balanced_score 
    return distribution_df, wasser_df


def figureMaker5(ax, distribution_df, wasser_df):
    """
    This makes figure 5.
    """
    # cartoon to show different shapes --> similar shapes
    i = 0
    ax[i].axis('off')

    i += 1
    ax[i].axis('off')

    i += 1

    sns.violinplot(x="Distribution type", y="G1 lifetime", hue="State", split=True, data=distribution_df, ax=ax[i])

    i += 1
    # state accuracy
    sns.regplot(x="Wasserstein distance", y="State Assignment Accuracy", data=wasser_df, label="tHMM", ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[0])
    sns.regplot(x="Wasserstein distance", y="KMeans accuracy", data=wasser_df, ax=ax[i], label="K-means", lowess=True, marker='+', scatter_kws=scatter_kws_list[1])
    ax[i].set_title("State Assignment Accuracy")
    ax[i].set_ylabel("Accuracy [%]")
    ax[i].set_ylim(bottom=10.0, top=101)
    ax[i].legend()
