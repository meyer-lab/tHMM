""" Figure 18 benchmark model with 5 states. """

import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import itertools as it
from .common import getSetup, subplotLabel, commonAnalyze, figureMaker
from ..LineageTree import LineageTree
from ..Analyze import Analyze_list
from ..Lineage_collections import Gem10uM

# # Gem10
# gem_tHMMobj_list, _ = Analyze_list([Gem10uM], 5, fpi=True)
# gem_states_list = [tHMMobj.predict() for tHMMobj in gem_tHMMobj_list]
# # assign the predicted states to each cell
# for idx, gem_tHMMobj in enumerate(gem_tHMMobj_list):
#     for lin_indx, lin in enumerate(gem_tHMMobj.X):
#         for cell_indx, cell in enumerate(lin.output_lineage):
#             cell.state = gem_states_list[idx][lin_indx][cell_indx]

# # create a pickle file for lapatinib
# pik1 = open("gem10.pkl", "wb")

# for laps in gem_tHMMobj_list:
#     pickle.dump(laps, pik1)
# pik1.close()

desired_num_cells = 31
num_data_points = 10
min_num_lineages = 5
max_num_lineages = 50

pik1 = open("gem10.pkl", "rb")
gem = []
for i in range(1):
    gem.append(pickle.load(pik1))

# T: transition probability matrix
T = gem[0].estimate.T
# pi: the initial probability vector
pi = gem[0].estimate.pi

E = gem[0].estimate.E

# Creating a list of populations to analyze over
num_lineages = np.linspace(min_num_lineages, max_num_lineages, num_data_points, dtype=int)
list_of_populations = []

for num in num_lineages:
    population = []

    nn = 0
    for _ in range(num):
        tmp_lineage = LineageTree.init_from_parameters(pi, T, E, desired_num_cells, censor_condition=3, desired_experiment_time=96)
        while len(tmp_lineage.output_lineage) < 3:
            tmp_lineage = LineageTree.init_from_parameters(pi, T, E, desired_num_cells, censor_condition=3, desired_experiment_time=96)
        population.append(tmp_lineage)
        nn += len(tmp_lineage.output_lineage)

    # Adding populations into a holder for analysing
    list_of_populations.append(population)


def makeFigure():
#     """
#     Makes figure S03.
#     """

#     # Get list of axis objects
    ax, f = getSetup((10, 14), (4, 3))

    figureMaker5(ax, *commonAnalyze(list_of_populations, num_states=5, parallel=False), num_lineages=num_lineages)

    subplotLabel(ax)

    return f

def figureMaker5(ax, x, paramEst, dictOut, paramTrues, num_lineages):
    """ Plot accuracy of state prediction and parameter estimation. """

    accuracies = dictOut['state_similarity']
    tr = dictOut['transition_matrix_similarity']
    pii = dictOut['pi_similarity']
    num_states = paramTrues.shape[1]
    for iii in dictOut["confusion_matrix"]:
        print(iii, "\n")
    print(paramTrues[:, :, 3])

    ### plot the distribution of Gamma G1 and SG2
    # create random variables for each state from their distribution using rvs
    obs_g1, obs_g2, g1b, g2b = [], [], [], []
    for s in range(num_states):
        obs = gem[0].estimate.E[s].rvs(size=100)
        obs_g1.append(obs[2])
        obs_g2.append(obs[3])
        g1b.append(gem[0].estimate.E[s].params[0])
        g2b.append(gem[0].estimate.E[s].params[1])

    # create the same-size state vector to use as hue
    sts_g = [100*[i+1] for i in range(num_states)]
    df = pd.DataFrame({'G1':list(it.chain(*obs_g1)), 'S-G2':list(it.chain(*obs_g2)), 'State':list(it.chain(*sts_g))})

    # plot distributions
    sns.kdeplot(data=df, x='G1', hue='State', fill=True, common_norm=False, alpha=0.5, linewidth=0, ax=ax[0])
    sns.kdeplot(data=df, x='S-G2', hue='State', fill=True, common_norm=False, alpha=0.5, linewidth=0, ax=ax[1])

    # plot bernoullis
    dfb = pd.DataFrame({'State' : ['1', '2', '3', '4', '5'],
                        'G1' : g1b,
                        'S-G2' : g2b})
    dfb[['State', 'G1', 'S-G2']].plot(x='State', kind='bar', ax=ax[2], rot=0)
    ax[2].set_ylabel("Division Probability")
    ax[2].set_ylim((0.0, 1.1))
    i = 2
    i += 1  # (b) bernoulli G1
    for j in range(num_states):
        sns.regplot(x=x, y=paramEst[:, j, 0], ax=ax[i], lowess=True, marker='+', color="C"+str(j), scatter_kws={"alpha": 0.5, "marker": "x", "s": 20, "color":"C"+str(j)})
        ax[i].axhline(paramTrues[0, j, 0], linestyle="--", c="C" + str(j), label="S "+str(j+1))
    ax[i].set_xlabel("Number of Cells")
    ax[i].set_ylabel("Bernoulli p")
    ax[i].set_title("Bernoulli p G1 Estimation")
    ax[i].set_ylim(bottom=0.5, top=1.05)
    ax[i].legend()

    i += 1 # (c) gamma shape G1
    for j in range(num_states):
        sns.regplot(x=x, y=paramEst[:, j, 2], ax=ax[i], lowess=True, marker='+', color="C" + str(j), scatter_kws={"alpha": 0.5, "marker": "x", "s": 20, "color":"C"+str(j)})
        ax[i].axhline(paramTrues[0, j, 2], linestyle="--", c="C"+str(j), label="S "+str(j+1))
    ax[i].set_ylabel(r"Gamma k")
    ax[i].set_title(r"Gamma k G1 Estimation")
    ax[i].set_ylim([0.0, max(paramTrues[0, :, 2])+2])
    ax[i].set_xlabel("Number of Cells")

    i += 1 # (d) gamma scale G1
    for j in range(num_states):
        sns.regplot(x=x, y=paramEst[:, j, 3], ax=ax[i], lowess=True, marker='+', color="C"+str(j), scatter_kws={"alpha": 0.5, "marker": "x", "s": 20, "color":"C"+str(j)})
        ax[i].axhline(paramTrues[0, j, 3], linestyle="--", c="C" + str(j), label="S "+str(j+1))
    ax[i].set_ylabel(r"Gamma $\theta$")
    ax[i].set_title(r"Gamma $\theta$ G1 Estimation")
    ax[i].set_ylim([0.0, max(paramTrues[0, :, 3]+15)])
    ax[i].set_xlabel("Number of Cells")
    ax[i].legend()

    i += 1  # (e) bernoulli G2
    for j in range(num_states):
        sns.regplot(x=x, y=paramEst[:, j, 1], ax=ax[i], lowess=True, marker='+', color="C"+str(j), scatter_kws={"alpha": 0.5, "marker": "x", "s": 20, "color":"C"+str(j)})
        ax[i].axhline(paramTrues[0, j, 1], linestyle="--", c="C" + str(j), label="S "+str(j+1))
    ax[i].set_xlabel("Number of Cells")
    ax[i].set_ylabel("Bernoulli p")
    ax[i].set_title("Bernoulli p G2 Estimation")
    ax[i].set_ylim(bottom=0.5, top=1.05)

    i += 1 # (f) gamma shape G2
    for j in range(num_states):
        sns.regplot(x=x, y=paramEst[:, j, 4], ax=ax[i], lowess=True, marker='+', color="C" + str(j), scatter_kws={"alpha": 0.5, "marker": "x", "s": 20, "color":"C"+str(j)})
        ax[i].axhline(paramTrues[0, j, 4], linestyle="--", c="C"+str(j), label="S "+str(j+1))
    ax[i].set_ylabel(r"Gamma k")
    ax[i].set_title(r"Gamma k G2 Estimation")
    ax[i].set_ylim([0.0, max(paramTrues[0, :, 4])+2])
    ax[i].set_xlabel("Number of Cells")

    i += 1 # (g) gamma scale G2
    for j in range(num_states):
        sns.regplot(x=x, y=paramEst[:, j, 5], ax=ax[i], lowess=True, marker='+', color="C"+str(j), scatter_kws={"alpha": 0.5, "marker": "x", "s": 20, "color":"C"+str(j)})
        ax[i].axhline(paramTrues[0, j, 5], linestyle="--", c="C" + str(j), label="S "+str(j+1))
    ax[i].set_ylabel(r"Gamma $\theta$")
    ax[i].set_title(r"Gamma $\theta$ G2 Estimation")
    ax[i].set_ylim([0.0, max(paramTrues[0, :, 5])+2])
    ax[i].set_xlabel("Number of Cells")

    i += 1 # (i) accuracy
    ax[i].set_ylim(bottom=0, top=101)
    sns.regplot(x=x, y=accuracies, ax=ax[i], lowess=True, color="C0", marker='+', scatter_kws={"alpha": 0.5, "marker": "x", "s": 20, "color":"C0"})
    ax[i].set_ylabel(r"Rand Index Accuracy [%]")
    ax[i].set_title("State Assignment Accuracy")
    ax[i].set_xlabel("Number of Cells")

    i += 1 # (j) T estimates
    ax[i].set_ylim(bottom=0, top=3.0)
    sns.regplot(x=x, y=tr, ax=ax[i], lowess=True, color="C0", marker='+', scatter_kws={"alpha": 0.5, "marker": "x", "s": 20, "color":"C0"})
    ax[i].set_ylabel(r"$||T-T_{est}||_{F}$")
    ax[i].set_title(r"Error in Estimating T")
    ax[i].set_xlabel("Number of Cells")

    i += 1 # (i) pi estimates
    ax[i].set_ylim(bottom=0, top=2.0)
    sns.regplot(x=num_lineages, y=pii, ax=ax[i], color="C0", lowess=True, marker='+', scatter_kws={"alpha": 0.5, "marker": "x", "s": 20, "color":"C0"})
    ax[i].set_ylabel(r"$||\pi-\pi_{est}||_{2}$")
    ax[i].set_title(r"Error in Estimating $\pi$")
    ax[i].set_xlabel("Number of Lineages")