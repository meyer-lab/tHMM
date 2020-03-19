"""
File: figure4.py
Purpose: Generates figure 4.
Figure 4 analyzes heterogeneous (2 state), pruned (by both time and fate), 
populations of lineages (more than one lineage per populations) 
with at least 16 cells per lineage 
over increasing number of lineages per population.
"""
import numpy as np

from .figureCommon import getSetup, subplotLabel, commonAnalyze, figureMaker, pi, T, E, desired_num_cells
from ..LineageTree import LineageTree


def makeFigure():
    """
    Makes figure 4.
    """

    # Get list of axis objects
    ax, f = getSetup((7, 6), (2, 3))

    figureMaker(ax, *accuracy_increased_cells())

    subplotLabel(ax)

    return f


def accuracy_increased_cells():
    """
    Calculates accuracy and parameter estimation 
    over an increasing number of lineages in a population for 
    a censored two-state model.
    We increase the desired number of cells in a lineage by
    the experiment time. 
    """

    # Creating a list of populations to analyze over
    num_lineages = np.linspace(1, 50, 50, dtype=int)
    list_of_populations = []
    for num in num_lineages:
        population = []
        
        for _ in range(num):
            # Creating a censored lineage
            tmp_lineage = LineageTree(pi, T, E, desired_num_cells, censor_condition=3, desired_experiment_time=experiment_time)
            
            while len(tmp_lineage.output_lineage) < 16:
                del tmp_lineage
                tmp_lineage = LineageTree(pi, T, E, desired_num_cells, censor_condition=3, desired_experiment_time=experiment_time)
            population.append(tmp_lineage)
        
        # Adding populations into a holder for analysing
        list_of_populations.append(population)

    return commonAnalyze(list_of_populations)
