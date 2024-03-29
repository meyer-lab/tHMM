"""
File: figureS05.py
Purpose: Generates figure S05.
Figure 05 analyzes heterogeneous (2 state), censored (by both time and fate),
single lineages (more than one lineage per population)
with different proportions of cells in states by
changing the values in the transition matrices.
"""

import numpy as np

from .common import (
    getSetup,
    subplotLabel,
    commonAnalyze,
    figureMaker,
    E,
    max_desired_num_cells,
    num_data_points,
)
from ..LineageTree import LineageTree
from ..BaumWelch import calculate_stationary


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
    list_of_Ts = [
        np.array([[i, 1.0 - i], [i, 1.0 - i]])
        for i in np.linspace(0.01, 0.99, num_data_points)
    ]
    list_of_Ts = [a + 5 * np.eye(2) for a in list_of_Ts]
    list_of_Ts = [a / np.sum(a, axis=1)[:, np.newaxis] for a in list_of_Ts]
    list_of_fpi = [calculate_stationary(a) for a in list_of_Ts]

    # generate lineages
    def genC(x):
        return LineageTree.rand_init(
            calculate_stationary(x),
            x,
            E,
            max_desired_num_cells,
            censor_condition=3,
            desired_experiment_time=700,
        )

    list_of_populations = [[genC(T) for _ in range(10)] for T in list_of_Ts]

    return commonAnalyze(
        list_of_populations,
        2,
        xtype="prop",
        list_of_fpi=list_of_fpi,
        list_of_fT=list_of_Ts,
    )
