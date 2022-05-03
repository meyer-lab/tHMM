"""
File: figureS03.py
Purpose: Generates figure S03.
Figure S03 analyzes heterogeneous (5 state), censored (by both time and fate),
populations of lineages (more than one lineage per populations).
"""
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
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

desired_num_cells = 15
num_data_points = 100
min_num_lineages = 25
max_num_lineages = 375

pik1 = open("gem10.pkl", "rb")
gem_tHMMobj_list = []
for i in range(1):
    gem_tHMMobj_list.append(pickle.load(pik1))

# T: transition probability matrix
T = gem_tHMMobj_list[0].estimate.T
# pi: the initial probability vector
pi = gem_tHMMobj_list[0].estimate.pi

E = gem_tHMMobj_list[0].estimate.E

# Creating a list of populations to analyze over
num_lineages = np.linspace(min_num_lineages, max_num_lineages, num_data_points, dtype=int)
list_of_populations = []

for num in num_lineages:
    population = []

    nn = 0
    for _ in range(num):
        tmp_lineage = LineageTree.init_from_parameters(pi, T, E, desired_num_cells, censor_condition=3, desired_experiment_time=96)
        population.append(tmp_lineage)
        nn += len(tmp_lineage.output_lineage)

    print(nn)
    # Adding populations into a holder for analysing
    list_of_populations.append(population)


def makeFigure():
#     """
#     Makes figure S03.
#     """

#     # Get list of axis objects
    ax, f = getSetup((10, 14), (4, 3))

    figureMaker5(ax, *commonAnalyze(list_of_populations, num_states=5, list_of_fpi=[pi]*len(list_of_populations), list_of_fT=[T]*len(list_of_populations), parallel=True), num_lineages=num_lineages)

    subplotLabel(ax)

    return f

def figureMaker5(ax, x, paramEst, dictOut, paramTrues, num_lineages):
    """ Plot accuracy of state prediction and parameter estimation. """

    accuracies = dictOut['state_similarity']
    tr = dictOut['transition_matrix_similarity']
    pii = dictOut['pi_similarity']
    num_states = paramTrues.shape[1]

    for i in range(3):
        ax[i].axis('off')

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
