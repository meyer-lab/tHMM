"""
File: figureS05.py
Purpose: Generates figure S05.
Figure 05 analyzes heterogeneous (2 state), censored (by both time and fate),
single lineages (more than one lineage per population)
with different proportions of cells in states by
changing the values in the transition matrices.
"""
import numpy as np

from .figureCommon import (
    getSetup,
    subplotLabel,
    commonAnalyze,
    figureMaker,
    pi,
    E,
    max_desired_num_cells,
    max_experiment_time,
    num_data_points,
)
from ..LineageTree import LineageTree


def makeFigure():
    """
    Makes figure 5.
    """

    # Get list of axis objects
    ax, f = getSetup((10, 10), (3, 3))

    figureMaker(ax, *accuracy(), xlabel=r"Cells in State 1 [$\%$]")

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
    for T in list_of_Ts:
        population = []

        for _ in range(4):
            tmp_lineage = LineageTree.init_from_parameters(pi, T, E, 0.6 * max_desired_num_cells, censor_condition=3, desired_experiment_time=max_experiment_time)
            if len(tmp_lineage.output_lineage) < 3:
                pass
            else:
                population.append(tmp_lineage)

        # Adding populations into a holder for analysing
        list_of_populations.append(population)
        list_of_fpi.append(pi)

    return commonAnalyze(list_of_populations, 2, xtype="prop", list_of_fpi=list_of_fpi)
