"""
File: figureS04.py
Purpose: Generates figure S04.
Figure S04 analyzes heterogeneous (2 state), NOT censored,
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
    num_data_points,
)
from ..LineageTree import LineageTree


def makeFigure():
    """
    Makes figure S04.
    """

    # Get list of axis objects
    ax, f = getSetup((10, 10), (3, 3))

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
    list_of_fpi = [pi] * len(list_of_Ts)

    # generate lineages
    def genF(x): return LineageTree.init_from_parameters(pi, x, E, max_desired_num_cells)
    list_of_populations = [[genF(T) for _ in range(10)] for T in list_of_Ts]

    return commonAnalyze(list_of_populations, 2, xtype="prop", list_of_fpi=list_of_fpi, list_of_Ts=list_of_Ts)
