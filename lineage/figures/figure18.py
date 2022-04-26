"""
File: figureS03.py
Purpose: Generates figure S03.
Figure S03 analyzes heterogeneous (5 state), censored (by both time and fate),
populations of lineages (more than one lineage per populations).
"""
import numpy as np
import pickle
import pandas as pd
import seaborn as sns
from .common import getSetup, subplotLabel, commonAnalyze
from ..LineageTree import LineageTree

pik1 = open("lapatinibs.pkl", "rb")
lapt_tHMMobj_list = []
for i in range(4):
    lapt_tHMMobj_list.append(pickle.load(pik1))


desired_num_cells = 15
num_data_points = 100
min_num_lineages = 25
max_num_lineages = 200

# T: transition probability matrix
T = lapt_tHMMobj_list[3].estimate.T
# pi: the initial probability vector
pi = lapt_tHMMobj_list[3].estimate.pi

E = lapt_tHMMobj_list[3].estimate.E

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
    """
    Makes figure S03.
    """

    # Get list of axis objects
    ax, f = getSetup((10, 14), (4, 3))

    figureMaker5(ax, *commonAnalyze(list_of_populations, num_states=6, list_of_fpi=[pi]*num_data_points, list_of_fT=[T]*num_data_points, parallel=True), num_lineages=num_lineages)

    subplotLabel(ax)

    return f

def figureMaker5(ax, x, paramEst, dictOut, paramTrues, num_lineages):
    """ Plot accuracy of state prediction and parameter estimation. """

    print("param trues", paramTrues.shape)
    accuracies = dictOut["state_similarity"]
    tr = dictOut["transition_matrix_similarity"]
    pii = dictOut["pi_similarity"]

    accuracy_df = pd.DataFrame(columns=["x", 'accuracy'])
    accuracy_df['x'] = x
    accuracy_df['accuracy'] = accuracies
    accuracy_df['tr'] = tr
    accuracy_df['pii'] = pii
    accuracy_df['bern 0 0'] = paramEst[:, 0, 0]  # bernoulli G1
    accuracy_df['bern 1 0'] = paramEst[:, 1, 0]
    accuracy_df['bern 2 0'] = paramEst[:, 2, 0]
    accuracy_df['bern 3 0'] = paramEst[:, 3, 0]
    accuracy_df['bern 4 0'] = paramEst[:, 4, 0]
    accuracy_df['bern 5 0'] = paramEst[:, 5, 0]

    accuracy_df['bern 0 1'] = paramEst[:, 0, 1]  # bernoulli G2
    accuracy_df['bern 1 1'] = paramEst[:, 1, 1]
    accuracy_df['bern 2 1'] = paramEst[:, 2, 1]
    accuracy_df['bern 3 1'] = paramEst[:, 3, 1]
    accuracy_df['bern 4 1'] = paramEst[:, 4, 1]
    accuracy_df['bern 5 1'] = paramEst[:, 5, 1]

    accuracy_df['0 0'] = paramEst[:, 0, 2]  # gamma shape G1
    accuracy_df['1 0'] = paramEst[:, 1, 2]
    accuracy_df['2 0'] = paramEst[:, 2, 2]
    accuracy_df['3 0'] = paramEst[:, 3, 2]
    accuracy_df['4 0'] = paramEst[:, 4, 2]
    accuracy_df['5 0'] = paramEst[:, 5, 2]

    accuracy_df['0 1'] = paramEst[:, 0, 3]  # gamma scale G1
    accuracy_df['1 1'] = paramEst[:, 1, 3]
    accuracy_df['2 1'] = paramEst[:, 2, 3]
    accuracy_df['3 1'] = paramEst[:, 3, 3]
    accuracy_df['4 1'] = paramEst[:, 4, 3]
    accuracy_df['5 1'] = paramEst[:, 5, 3]

    accuracy_df['0 2'] = paramEst[:, 0, 4]  # gamma shape G2
    accuracy_df['1 2'] = paramEst[:, 1, 4]
    accuracy_df['2 2'] = paramEst[:, 2, 4]
    accuracy_df['3 2'] = paramEst[:, 3, 4]
    accuracy_df['4 2'] = paramEst[:, 4, 4]
    accuracy_df['5 2'] = paramEst[:, 5, 4]

    accuracy_df['0 3'] = paramEst[:, 0, 5]  # gamma scale G2
    accuracy_df['1 3'] = paramEst[:, 1, 5]
    accuracy_df['2 3'] = paramEst[:, 2, 5]
    accuracy_df['3 3'] = paramEst[:, 3, 5]
    accuracy_df['4 3'] = paramEst[:, 4, 5]
    accuracy_df['5 3'] = paramEst[:, 5, 5]

    accuracy_df['num lineages'] = num_lineages

    for i in range(3):
        ax[i].axis('off')

    i += 1  # (b) bernoulli G1
    for j in range(6):
        sns.regplot(x="x", y="bern "+str(j) + " 0", data=accuracy_df, ax=ax[i], lowess=True, marker='+', color="C"+str(j), scatter_kws={"alpha": 0.5, "marker": "x", "s": 20, "color":"C"+str(j)})
        ax[i].axhline(y=paramTrues[1, j, 0], linestyle="--", c="C" + str(j), label="S "+str(j+1))
    ax[i].set_xlabel("Number of Cells")
    ax[i].set_ylabel("Bernoulli p")
    ax[i].set_title("Bernoulli p G1 Estimation")
    ax[i].set_ylim(bottom=0.5, top=1.05)
    ax[i].legend()

    i += 1 # (c) gamma shape G1
    for j in range(6):
        sns.regplot(x="x", y=str(j) + " 0", data=accuracy_df, ax=ax[i], lowess=True, marker='+', color="C" + str(j), scatter_kws={"alpha": 0.5, "marker": "x", "s": 20, "color":"C"+str(j)})
        ax[i].axhline(y=paramTrues[1, j, 2], linestyle="--", c="C"+str(j), label="S "+str(j+1))
    ax[i].set_ylabel(r"Gamma k")
    ax[i].set_title(r"Gamma k G1 Estimation")
    ax[i].set_ylim([0.0, 60.0])
    ax[i].set_xlabel("Number of Cells")

    i += 1 # (d) gamma scale G1
    for j in range(6):
        sns.regplot(x="x", y=str(j) + " 1", data=accuracy_df, ax=ax[i], lowess=True, marker='+', color="C"+str(j), scatter_kws={"alpha": 0.5, "marker": "x", "s": 20, "color":"C"+str(j)})
        ax[i].axhline(y=paramTrues[1, j, 3], linestyle="--", c="C" + str(j), label="S "+str(j+1))
    ax[i].set_ylabel(r"Gamma $\theta$")
    ax[i].set_title(r"Gamma $\theta$ G1 Estimation")
    ax[i].set_ylim([0.0, 80.0])
    ax[i].set_xlabel("Number of Cells")
    ax[i].legend()

    i += 1  # (e) bernoulli G2
    for j in range(6):
        sns.regplot(x="x", y="bern "+str(j) + " 1", data=accuracy_df, ax=ax[i], lowess=True, marker='+', color="C"+str(j), scatter_kws={"alpha": 0.5, "marker": "x", "s": 20, "color":"C"+str(j)})
        ax[i].axhline(y=paramTrues[1, j, 1], linestyle="--", c="C" + str(j), label="S "+str(j+1))
    ax[i].set_xlabel("Number of Cells")
    ax[i].set_ylabel("Bernoulli p")
    ax[i].set_title("Bernoulli p G2 Estimation")
    ax[i].set_ylim(bottom=0.5, top=1.05)

    i += 1 # (f) gamma shape G2
    for j in range(6):
        sns.regplot(x="x", y=str(j) + " 2", data=accuracy_df, ax=ax[i], lowess=True, marker='+', color="C" + str(j), scatter_kws={"alpha": 0.5, "marker": "x", "s": 20, "color":"C"+str(j)})
        ax[i].axhline(y=paramTrues[1, j, 4], linestyle="--", c="C"+str(j), label="S "+str(j+1))
    ax[i].set_ylabel(r"Gamma k")
    ax[i].set_title(r"Gamma k G2 Estimation")
    # ax[i].set_ylim([0.0, 85.0])
    ax[i].set_xlabel("Number of Cells")

    i += 1 # (g) gamma scale G2
    for j in range(6):
        sns.regplot(x="x", y=str(j) + " 3", data=accuracy_df, ax=ax[i], lowess=True, marker='+', color="C"+str(j), scatter_kws={"alpha": 0.5, "marker": "x", "s": 20, "color":"C"+str(j)})
        ax[i].axhline(y=paramTrues[1, j, 5], linestyle="--", c="C" + str(j), label="S "+str(j+1))
    ax[i].set_ylabel(r"Gamma $\theta$")
    ax[i].set_title(r"Gamma $\theta$ G2 Estimation")
    # ax[i].set_ylim([0.0, 10.0])
    ax[i].set_xlabel("Number of Cells")

    i += 1 # (i) accuracy
    ax[i].set_ylim(bottom=0, top=101)
    sns.regplot(x="x", y="accuracy", data=accuracy_df, ax=ax[i], lowess=True, color="C0", marker='+', scatter_kws={"alpha": 0.5, "marker": "x", "s": 20, "color":"C"+str(j)})
    ax[i].set_ylabel(r"Rand Index Accuracy [%]")
    ax[i].set_title("State Assignment Accuracy")
    ax[i].set_xlabel("Number of Cells")

    i += 1 # (j) T estimates
    ax[i].set_ylim(bottom=0, top=3.0)
    sns.regplot(x="x", y="tr", data=accuracy_df, ax=ax[i], lowess=True, color="C0", marker='+', scatter_kws={"alpha": 0.5, "marker": "x", "s": 20, "color":"C"+str(j)})
    ax[i].set_ylabel(r"$||T-T_{est}||_{F}$")
    ax[i].set_title(r"Error in Estimating T")
    ax[i].set_xlabel("Number of Cells")

    i += 1 # (i) pi estimates
    ax[i].set_ylim(bottom=0, top=2.0)
    sns.regplot(x="num lineages", y="pii", data=accuracy_df, ax=ax[i], color="C0", lowess=True, marker='+', scatter_kws={"alpha": 0.5, "marker": "x", "s": 20, "color":"C"+str(j)})
    ax[i].set_ylabel(r"$||\pi-\pi_{est}||_{2}$")
    ax[i].set_title(r"Error in Estimating $\pi$")
    ax[i].set_xlabel("Number of Lineages")
