"""
File: figure5.py
Purpose: Generates figure 5.
Figure 5 analyzes heterogeneous (2 state), NOT censored,
single lineages (no more than one lineage per population)
with different proportions of cells in states by
changing the values in the transition matrices.
Includes G1 and G2 phases separately.
"""
from string import ascii_lowercase
from cycler import cycler
import numpy as np
from matplotlib import gridspec, pyplot as plt
import seaborn as sns
from ..Analyze import run_Results_over, run_Analyze_over
from ..states.StateDistPhase import StateDistribution2
from ..LineageTree import LineageTree
from .figureCommon import lineage_good_to_analyze, getSetup, commonAnalyze, subplotLabel

# pi: the initial probability vector
pi = np.array([0.5, 0.5], dtype="float")

# T: transition probability matrix
T = np.array([[0.9, 0.1], [0.1, 0.9]], dtype="float")

# bern, gamma_a, gamma_scale
state0 = StateDistribution2(0.99, 20, 3, 15, 5)
state1 = StateDistribution2(0.75, 17, 1, 12, 4)
E = [state0, state1]


min_desired_num_cells = (2**8) - 1
max_desired_num_cells = (2**9) - 1

min_min_lineage_length = 10

min_experiment_time = 72
max_experiment_time = 144

min_num_lineages = 1
max_num_lineages = 100

num_data_points = 50


def figureMaker(ax, x, paramEst, accuracies, tr, pii, paramTrues, xlabel="Number of Cells"):
    """
    Makes the common 6 panel figures displaying parameter estimation across lineages
    of various types and sizes.
    """
    i = 0
    ax[i].set_xlabel(xlabel)
    ax[i].scatter(x, paramEst[:, 0, 0], edgecolors="k", marker="o", alpha=0.5)
    ax[i].scatter(x, paramEst[:, 1, 0], edgecolors="k", marker="o", alpha=0.5)
    ax[i].set_ylim(bottom=0, top=1.02)
    ax[i].set_ylabel("Bernoulli $p$")
    ax[i].scatter(x, paramTrues[:, 0, 0], marker="_", alpha=0.5)
    ax[i].scatter(x, paramTrues[:, 1, 0], marker="_", alpha=0.5)
    ax[i].set_title(r"Bernoulli $p$")
    ax[i].grid(linestyle="--")
    ax[i].tick_params(axis="both", which="major", grid_alpha=0.25)

    i += 1
    ax[i].set_xlabel(xlabel)
    ax[i].scatter(x, paramEst[:, 0, 1], edgecolors="k", marker="o", alpha=0.5)
    ax[i].scatter(x, paramEst[:, 1, 1], edgecolors="k", marker="o", alpha=0.5)
    ax[i].set_ylabel(r"Gamma $k$")
    ax[i].scatter(x, paramTrues[:, 0, 1], marker="_", alpha=0.5)
    ax[i].scatter(x, paramTrues[:, 1, 1], marker="_", alpha=0.5)
    ax[i].set_title(r"Gamma $k$ G1")
    ax[i].grid(linestyle="--")
    ax[i].tick_params(axis="both", which="major", grid_alpha=0.25)

    i += 1
    ax[i].set_xlabel(xlabel)
    ax[i].scatter(x, paramEst[:, 0, 2], edgecolors="k", marker="o", alpha=0.5)
    ax[i].scatter(x, paramEst[:, 1, 2], edgecolors="k", marker="o", alpha=0.5)
    ax[i].set_ylabel(r"Gamma $\theta$")
    ax[i].scatter(x, paramTrues[:, 0, 2], marker="_", alpha=0.5, label="State 1")
    ax[i].scatter(x, paramTrues[:, 1, 2], marker="_", alpha=0.5, label="State 2")
    ax[i].legend()
    ax[i].set_title(r"Gamma $\theta$ G1")
    ax[i].grid(linestyle="--")
    ax[i].tick_params(axis="both", which="major", grid_alpha=0.25)

    i += 1
    ax[i].set_xlabel(xlabel)
    ax[i].scatter(x, paramEst[:, 0, 3], edgecolors="k", marker="o", alpha=0.5)
    ax[i].scatter(x, paramEst[:, 1, 3], edgecolors="k", marker="o", alpha=0.5)
    ax[i].set_ylabel(r"Gamma $k$")
    ax[i].scatter(x, paramTrues[:, 0, 3], marker="_", alpha=0.5)
    ax[i].scatter(x, paramTrues[:, 1, 3], marker="_", alpha=0.5)
    ax[i].set_title(r"Gamma $k$ G2")
    ax[i].grid(linestyle="--")
    ax[i].tick_params(axis="both", which="major", grid_alpha=0.25)

    i += 1
    ax[i].set_xlabel(xlabel)
    ax[i].scatter(x, paramEst[:, 0, 4], edgecolors="k", marker="o", alpha=0.5)
    ax[i].scatter(x, paramEst[:, 1, 4], edgecolors="k", marker="o", alpha=0.5)
    ax[i].set_ylabel(r"Gamma $\theta$")
    ax[i].scatter(x, paramTrues[:, 0, 4], marker="_", alpha=0.5, label="State 1")
    ax[i].scatter(x, paramTrues[:, 1, 4], marker="_", alpha=0.5, label="State 2")
    ax[i].legend()
    ax[i].set_title(r"Gamma $\theta$ G2")
    ax[i].grid(linestyle="--")
    ax[i].tick_params(axis="both", which="major", grid_alpha=0.25)

    i += 1
    ax[i].set_xlabel(xlabel)
    ax[i].set_ylim(bottom=0, top=101)
    ax[i].scatter(x, accuracies, c="k", marker="o", label="Accuracy", edgecolors="k", alpha=0.25)
    ax[i].set_ylabel(r"Accuracy [\%]")
    ax[i].axhline(y=100, linestyle="--", linewidth=2, color="k", alpha=1)
    ax[i].set_title("State Assignment Accuracy")
    ax[i].grid(linestyle="--")
    ax[i].tick_params(axis="both", which="major", grid_alpha=0.25)

    i += 1
    ax[i].set_xlabel(xlabel)
    ax[i].set_ylim(bottom=0, top=max(tr) + 0.2)
    ax[i].scatter(x, tr, c="k", marker="o", edgecolors="k", alpha=0.25)
    ax[i].set_ylabel(r"$||T-T_{est}||_{F}$")
    ax[i].axhline(y=0, linestyle="--", linewidth=2, color="k", alpha=1)
    ax[i].set_title("Transition Matrix Estimation")
    ax[i].grid(linestyle="--")
    ax[i].tick_params(axis="both", which="major", grid_alpha=0.25)

    i += 1
    ax[i].set_xlabel(xlabel)
    ax[i].set_ylim(bottom=0, top=max(pii) + 0.2)
    ax[i].scatter(x, pii, c="k", marker="o", edgecolors="k", alpha=0.25)
    ax[i].set_ylabel(r"$||\pi-\pi_{est}||_{2}$")
    ax[i].axhline(y=0, linestyle="--", linewidth=2, color="k", alpha=1)
    ax[i].set_title("Initial Probability Matrix Estimation")
    ax[i].grid(linestyle="--")
    ax[i].tick_params(axis="both", which="major", grid_alpha=0.25)

def makeFigure():
    """
    Makes figure 16.
    """

    # Get list of axis objects
    ax, f = getSetup((12, 6), (2, 4))

    figureMaker(ax, *accuracy(), xlabel=r"Cells in State 0 [$\%$]")

    subplotLabel(ax)

    return f


def accuracy():
    """
    Calculates accuracy and parameter estimation
    over an similar number of cells in a lineage for
    a uncensored two-state model but differing state distribution.
    We increase the proportion of cells in a lineage by
    fixing the Transition matrix to be biased towards state 0.
    """

    # Creating a list of populations to analyze over
    list_of_Ts = [np.array([[i, 1.0 - i], [i, 1.0 - i]]) for i in np.linspace(0.1, 0.9, num_data_points)]
    list_of_populations = []
    list_of_fpi = []
    list_of_fT = []
    list_of_fE = []
    for T in list_of_Ts:
        population = []

        good2go = False
        while not good2go:
            tmp_lineage = LineageTree(pi, T, E, max_desired_num_cells)
            good2go = lineage_good_to_analyze(tmp_lineage)

        population.append(tmp_lineage)

        # Adding populations into a holder for analysing
        list_of_populations.append(population)
        list_of_fpi.append(pi)
        list_of_fT.append(T)
        list_of_fE.append(E)

    return commonAnalyze(list_of_populations, xtype="prop", list_of_fpi=list_of_fpi)
