""" This file contains functions for plotting different phenotypes in the manuscript. """

import pandas as pd
import seaborn as sns
from .figureCommon import (
    getSetup,
    subplotLabel,
    pi,
    T,
    E2
)
from ..LineageTree import LineageTree
from ..Analyze import Analyze, Results


def makeFigure():
    """
    Makes fig 2.
    """

    # Get list of axis objects
    ax, f = getSetup((10.0, 7.5), (3, 4))

    figureMaker2(ax, *forHistObs([LineageTree.init_from_parameters(pi, T, E2, desired_num_cells=2**8 - 1)]))

    subplotLabel(ax)

    return f


def forHistObs(X):
    """ To plot the histogram of the observations regardless of their state.

    :param X: list of lineages in the population.
    :type X: list
    """

    results_dict = Results(*Analyze(X, 2))
    pred_states_by_lineage = results_dict["switched_pred_states_by_lineage"][0]  # only one lineage

    BernoulliG1_hist = pd.DataFrame(columns=["States", "Count", "Fate"])
    BernoulliG1_hist["States"] = ["State 1"] + ["State 1"] +\
                                 ["State 2"] + ["State 2"] +\
                                 ["Cumulative"] + ["Cumulative"]
    BernoulliG1_hist["Count"] = [sum([1 for lineage in X for cell in lineage.output_lineage if cell.obs[0] == 0 and cell.state == 0])] +\
                                [sum([1 for lineage in X for cell in lineage.output_lineage if cell.obs[0] == 1 and cell.state == 0])] +\
                                [sum([1 for lineage in X for cell in lineage.output_lineage if cell.obs[0] == 0 and cell.state == 1])] +\
                                [sum([1 for lineage in X for cell in lineage.output_lineage if cell.obs[0] == 1 and cell.state == 1])] +\
                                [sum([1 for lineage in X for cell in lineage.output_lineage if cell.obs[0] == 0])] +\
                                [sum([1 for lineage in X for cell in lineage.output_lineage if cell.obs[0] == 1])]
    BernoulliG1_hist["Fate"] = ["Death"] + ["Transition to G2"] +\
                              ["Death"] + ["Transition to G2"] +\
                              ["Death"] + ["Transition to G2"]

    BernoulliG2_hist = pd.DataFrame(columns=["States", "Count", "Fate"])
    BernoulliG2_hist["States"] = BernoulliG1_hist["States"]
    BernoulliG2_hist["Count"] = [sum([1 for lineage in X for cell in lineage.output_lineage if cell.obs[1] == 0 and cell.state == 0])] +\
                                [sum([1 for lineage in X for cell in lineage.output_lineage if cell.obs[1] == 1 and cell.state == 0])] +\
                                [sum([1 for lineage in X for cell in lineage.output_lineage if cell.obs[1] == 0 and cell.state == 1])] +\
                                [sum([1 for lineage in X for cell in lineage.output_lineage if cell.obs[1] == 1 and cell.state == 1])] +\
                                [sum([1 for lineage in X for cell in lineage.output_lineage if cell.obs[1] == 0])] +\
                                [sum([1 for lineage in X for cell in lineage.output_lineage if cell.obs[1] == 1])]
    BernoulliG2_hist["Fate"] = ["Death"] + ["Division"] +\
                              ["Death"] + ["Division"] +\
                              ["Death"] + ["Division"]

    BernoulliG1_hist_est = pd.DataFrame(columns=["States", "Count", "Fate"])
    BernoulliG1_hist_est["States"] = BernoulliG1_hist["States"]
    BernoulliG1_hist_est["Count"] = [sum([1 for lineage in X for idx, cell in enumerate(lineage.output_lineage) if cell.obs[0] == 0 and pred_states_by_lineage[idx] == 0])] +\
                                    [sum([1 for lineage in X for idx, cell in enumerate(lineage.output_lineage) if cell.obs[0] == 1 and pred_states_by_lineage[idx] == 0])] +\
                                    [sum([1 for lineage in X for idx, cell in enumerate(lineage.output_lineage) if cell.obs[0] == 0 and pred_states_by_lineage[idx] == 1])] +\
                                    [sum([1 for lineage in X for idx, cell in enumerate(lineage.output_lineage) if cell.obs[0] == 1 and pred_states_by_lineage[idx] == 1])] +\
                                    [sum([1 for lineage in X for idx, cell in enumerate(lineage.output_lineage) if cell.obs[0] == 0])] +\
                                    [sum([1 for lineage in X for idx, cell in enumerate(lineage.output_lineage) if cell.obs[0] == 1])]
    BernoulliG1_hist_est["Fate"] = BernoulliG1_hist["Fate"]

    BernoulliG2_hist_est = pd.DataFrame(columns=["States", "Count", "Fate"])
    BernoulliG2_hist_est["States"] = BernoulliG1_hist["States"]
    BernoulliG2_hist_est["Count"] = [sum([1 for lineage in X for idx, cell in enumerate(lineage.output_lineage) if cell.obs[1] == 0 and pred_states_by_lineage[idx] == 0])] +\
                                    [sum([1 for lineage in X for idx, cell in enumerate(lineage.output_lineage) if cell.obs[1] == 1 and pred_states_by_lineage[idx] == 0])] +\
                                    [sum([1 for lineage in X for idx, cell in enumerate(lineage.output_lineage) if cell.obs[1] == 0 and pred_states_by_lineage[idx] == 1])] +\
                                    [sum([1 for lineage in X for idx, cell in enumerate(lineage.output_lineage) if cell.obs[1] == 1 and pred_states_by_lineage[idx] == 1])] +\
                                    [sum([1 for lineage in X for idx, cell in enumerate(lineage.output_lineage) if cell.obs[1] == 0])] +\
                                    [sum([1 for lineage in X for idx, cell in enumerate(lineage.output_lineage) if cell.obs[1] == 1])]
    BernoulliG2_hist_est["Fate"] = BernoulliG2_hist["Fate"]

    # state 1 observations
    obsGammaG1S1 = [cell.obs[2] for lineage in X for cell in lineage.output_lineage if cell.state == 0]
    obsGammaG1S2 = [cell.obs[2] for lineage in X for cell in lineage.output_lineage if cell.state == 1]
    obsGammaG1 = [cell.obs[2] for lineage in X for cell in lineage.output_lineage]

    obsGammaG1S1_est = [cell.obs[2] for lineage in X for idx, cell in enumerate(lineage.output_lineage) if pred_states_by_lineage[idx] == 0]
    obsGammaG1S2_est = [cell.obs[2] for lineage in X for idx, cell in enumerate(lineage.output_lineage) if pred_states_by_lineage[idx] == 1]
    obsGammaG1_est = [cell.obs[2] for lineage in X for idx, cell in enumerate(lineage.output_lineage)]

    # state 2 observations
    obsGammaG2S1 = [cell.obs[3] for lineage in X for cell in lineage.output_lineage if cell.state == 0]
    obsGammaG2S2= [cell.obs[3] for lineage in X for cell in lineage.output_lineage if cell.state == 1]
    obsGammaG2= [cell.obs[3] for lineage in X for cell in lineage.output_lineage]

    obsGammaG2S1_est= [cell.obs[3] for lineage in X for idx, cell in enumerate(lineage.output_lineage) if pred_states_by_lineage[idx] == 0]
    obsGammaG2S2_est= [cell.obs[3] for lineage in X for idx, cell in enumerate(lineage.output_lineage) if pred_states_by_lineage[idx] == 1]
    obsGammaG2_est= [cell.obs[3] for lineage in X for idx, cell in enumerate(lineage.output_lineage)]


    GammaG1_hist= pd.DataFrame(columns=['States', 'Time spent in G1'])
    GammaG1_hist['Time spent in G1']= obsGammaG1S1 + obsGammaG1S2 + obsGammaG1
    GammaG1_hist['States']= ['State 1'] * len(obsGammaG1S1) + ['State 2'] * len(obsGammaG1S2) + ['Total'] * len(obsGammaG1)

    GammaG2_hist= pd.DataFrame(columns=['States', 'Time spent in G1'])
    GammaG2_hist['Time spent in G2']= obsGammaG2S1 + obsGammaG2S2 + obsGammaG2
    GammaG2_hist['States']= ['State 1'] * len(obsGammaG2S1) + ['State 2'] * len(obsGammaG2S2) + ['Total'] * len(obsGammaG2)

    GammaG1_hist_est= pd.DataFrame(columns=['States', 'Time spent in G1'])
    GammaG1_hist_est['Time spent in G1']= obsGammaG1S1_est + obsGammaG1S2_est + obsGammaG1_est
    GammaG1_hist_est['States']= ['State 1'] * len(obsGammaG1S1_est) + ['State 2'] * len(obsGammaG1S2_est) + ['Total'] * len(obsGammaG1_est)

    GammaG2_hist_est= pd.DataFrame(columns=['States', 'Time spent in G2'])
    GammaG2_hist_est['Time spent in G2']= obsGammaG2S1_est + obsGammaG2S2_est + obsGammaG2_est
    GammaG2_hist_est['States']= ['State 1'] * len(obsGammaG2S1_est) + ['State 2'] * len(obsGammaG2S2_est) + ['Total'] * len(obsGammaG2_est)

    return BernoulliG1_hist, GammaG1_hist, BernoulliG2_hist, GammaG2_hist, BernoulliG1_hist_est, GammaG1_hist_est, BernoulliG2_hist_est, GammaG2_hist_est


def figureMaker2(ax, BernoulliG1_hist, GammaG1_hist, BernoulliG2_hist, GammaG2_hist, BernoulliG1_hist_est, GammaG1_hist_est, BernoulliG2_hist_est, GammaG2_hist_est):
    """
    Makes the common 6 panel figures displaying parameter estimation across lineages
    of various types and sizes.
    """
    i= 0
    ax[i].axis('off')

    i += 1
    ax[i].axis('off')

    i += 1
    ax[i].axis('off')

    i += 1
    ax[i].axis('off')

    i += 1
    sns.barplot(x="States", y="Count", hue="Fate", data=BernoulliG1_hist, ax=ax[i])
    ax[i].set_xlabel("")
    ax[i].set_ylabel("Bernoulli distribution")
    ax[i].set_title(r"Fate after G1")
    ax[i].set_ylim(0,250)

    i += 1
    sns.violinplot(x="States", y="Time spent in G1", data=GammaG1_hist, ax=ax[i], scale="count", inner="quartile")
    ax[i].set_xlabel("")
    ax[i].set_ylabel(r"Gamma distribution")
    ax[i].set_title(r"Time spent in G1")

    i += 1
    sns.barplot(x="States", y="Count", hue="Fate", data=BernoulliG2_hist, ax=ax[i])
    ax[i].set_xlabel("")
    ax[i].set_ylabel("Bernoulli distribution")
    ax[i].set_title(r"Fate after G2")
    ax[i].set_ylim(0,250)

    i += 1
    sns.violinplot(x="States", y="Time spent in G2", data=GammaG2_hist, ax=ax[i], scale="count", inner="quartile")
    ax[i].set_xlabel("")
    ax[i].set_ylabel(r"Gamma distribution")
    ax[i].set_title(r"Time spent in G2")

    i += 1
    sns.barplot(x="States", y="Count", hue="Fate", data=BernoulliG1_hist_est, ax=ax[i])
    ax[i].set_xlabel("")
    ax[i].set_ylabel("Bernoulli distribution")
    ax[i].set_title(r"Fate after G1")
    ax[i].set_ylim(0,250)

    i += 1
    sns.violinplot(x="States", y="Time spent in G1", data=GammaG1_hist_est, ax=ax[i], scale="count", inner="quartile")
    ax[i].set_xlabel("")
    ax[i].set_ylabel(r"Gamma distribution")
    ax[i].set_title(r"Time spent in G1")

    i += 1
    sns.barplot(x="States", y="Count", hue="Fate", data=BernoulliG2_hist_est, ax=ax[i])
    ax[i].set_xlabel("")
    ax[i].set_ylabel("Bernoulli distribution")
    ax[i].set_title(r"Fate after G2")
    ax[i].set_ylim(0,250)

    i += 1
    sns.violinplot(x="States", y="Time spent in G2", data=GammaG2_hist_est, ax=ax[i], scale="count", inner="quartile")
    ax[i].set_xlabel("")
    ax[i].set_ylabel(r"Gamma distribution")
    ax[i].set_title(r"Time spent in G2")
