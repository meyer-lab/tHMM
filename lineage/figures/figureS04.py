"""
File: figureS04.py
Purpose: Generates figure S04.
Figure S04 analyzes heterogeneous (2 state), NOT censored,
single lineages (more than one lineage per population)
with different proportions of cells in states by
changing the values in the transition matrices.
"""

from ..BaumWelch import calculate_stationary
from ..LineageTree import LineageTree
from .common import (
    E,
    commonAnalyze,
    figureMaker,
    getSetup,
    make_transition_matrix_sweep,
    max_desired_num_cells,
    subplotLabel,
)


def makeFigure():
    """
    Makes figure S04.
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
    list_of_Ts, list_of_fpi = make_transition_matrix_sweep()

    # generate lineages
    def genF(x):
        return LineageTree.rand_init(calculate_stationary(x), x, E, max_desired_num_cells)

    list_of_populations = [[genF(T) for _ in range(10)] for T in list_of_Ts]

    return commonAnalyze(
        list_of_populations,
        2,
        xtype="prop",
        list_of_fpi=list_of_fpi,
        list_of_fT=list_of_Ts,
    )
