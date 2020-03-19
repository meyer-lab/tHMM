"""
File: figure4.py
Purpose: Generates figure 4.
Figure 4 analyzes heterogeneous (2 state), pruned (by both time and fate), populations of lineages
(more than one lineage per populations) with at least 10 cells per lineage over increasing
number of lineages per population.
"""
import numpy as np

from .figureCommon import getSetup, moving_average, subplotLabel
from ..Analyze import run_Analyze_over, run_Results_over
from ..LineageTree import LineageTree
from ..StateDistribution import StateDistribution


state_obj0 = StateDistribution(0.99, 20, 5)
state_obj1 = StateDistribution(0.88, 10, 1)
E = [state_obj0, state_obj1]

# pi: the initial probability vector
piiii = np.array([0.6, 0.4])

# T: transition probability matrix
T = np.array([[0.75, 0.25],
              [0.15, 0.85]])


def makeFigure():
    """
    Makes figure 4.
    """

    # Get list of axis objects
    ax, f = getSetup((7, 6), (2, 3))

    figure_maker(ax, *accuracy_increased_cells())

    subplotLabel(ax)

    return f


def accuracy_increased_cells():
    """
    Calculates parameter estimation by increasing the number of cells in a lineage for a two-state model.
    """

    desired_num_cells = 2**9 - 1
    experiment_time = 50

    # Creating a list of populations to analyze over
    num_lineages = list(range(1, 50))
    list_of_populations = []
    for num in num_lineages:
        population = []
        for _ in range(num):
            # Creating an unpruned and pruned lineage
            tmp_lineage = LineageTree(piiii, T, E, desired_num_cells, censor_condition=3, desired_experiment_time=experiment_time)
            if len(tmp_lineage.output_lineage) < 10:
                del tmp_lineage
                tmp_lineage = LineageTree(piiii, T, E, desired_num_cells, censor_condition=3, desired_experiment_time=experiment_time)
            population.append(tmp_lineage)
        # Adding populations into a holder for analysing
        list_of_populations.append(population)

    return figFourCommon(list_of_populations)




