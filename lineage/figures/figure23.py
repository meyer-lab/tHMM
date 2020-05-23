"""
File: figure23.py
Purpose: Generates figure 23.
Figure 23 analyzes heterogeneous (2 state), uncensored,
populations of lineages (more than one lineage per populations).
This particular state distribution has phases.
"""
import numpy as np

from .figureCommon import (
    getSetup,
    subplotLabel,
    commonAnalyze,
    pi,
    T,
    E2,
    min_desired_num_cells,
    lineage_good_to_analyze,
    min_num_lineages,
    max_num_lineages,
    num_data_points,
)
from ..LineageTree import LineageTree


def makeFigure():
    """
    Makes figure 23.
    """

    # Get list of axis objects
    ax, f = getSetup((9.333, 6), (2, 4))

    figureMaker2(ax, *accuracy())

    subplotLabel(ax)

    return f


def accuracy():
    """
    Calculates accuracy and parameter estimation
    over an increasing number of lineages in a population for
    a uncensored two-state model.
    We increase the desired number of cells in a lineage by
    the experiment time.
    """

    # Creating a list of populations to analyze over
    num_lineages = np.linspace(min_num_lineages, max_num_lineages, num_data_points, dtype=int)
    list_of_populations = []
    list_of_fpi = []
    list_of_fT = []
    list_of_fE = []
    for num in num_lineages:
        population = []

        for _ in range(num):

            good2go = False
            while not good2go:
                tmp_lineage = LineageTree(pi, T, E2, min_desired_num_cells)
                good2go = lineage_good_to_analyze(tmp_lineage)

            population.append(tmp_lineage)

        # Adding populations into a holder for analysing
        list_of_populations.append(population)
        list_of_fpi.append(pi)
        list_of_fT.append(T)
        list_of_fE.append(E2)

    return commonAnalyze(list_of_populations)

def figureMaker2(ax, x, paramEst, accuracies, tr, pii, paramTrues, xlabel="Number of Cells"):
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