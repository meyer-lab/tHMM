"""
File: figureS12.py
Purpose: Generates figure S12.
Figure S12 shows the accuracy of state assignment versus 
increasing cell numbers for 2-state, 3-state, and 4-state models.
"""
import numpy as np
from ..Analyze import Analyze, Results

from .figureCommon import (
    getSetup,
    subplotLabel,
    commonAnalyze,
    figureMaker,
    pi,
    T,
    E,
    min_desired_num_cells,
    lineage_good_to_analyze,
    min_num_lineages,
    max_num_lineages,
    num_data_points,
)
from ..LineageTree import LineageTree


def makeFigure():
    """
    Makes figure S12.
    """

    # Get list of axis objects
    ax, f = getSetup((10, 10), (1, 3))

    figureMaker(ax, *accuracy())

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

    self.X = [LineageTree.init_from_parameters(pi, T, E, desired_num_cells=(2 ** 11) - 1)]
    self.pi = np.array([0.55, 0.35, 0.10])
    self.T = np.array([[0.75, 0.20, 0.05], [0.1, 0.85, 0.05], [0.1, 0.1, 0.8]])

    X2 = []
    X3 = []
    X4 = []
        for _ in range(10):
            good2go = False
            while not good2go:
                tmp_lineage = LineageTree.init_from_parameters(self.pi, self.T, self.E, 2 ** 11 - 1)
                good2go = lineage_good_to_analyze(tmp_lineage)

        X2.append(tmp_lineage)
        tree_obj, predicted_states, LL = Analyze(X, 2)
        results_dict = Results(tree_obj, predicted_states, LL)
        accuracy = results_dict["accuracy_after_switching"]
        
        X3.append(tmp_lineage)
        tree_obj, predicted_states, LL = Analyze(X, 3)
        results_dict = Results(tree_obj, predicted_states, LL)
        accuracy = results_dict["accuracy_after_switching"]
        
        X4.append(tmp_lineage)
        tree_obj, predicted_states, LL = Analyze(X, 4)
        results_dict = Results(tree_obj, predicted_states, LL)
        accuracy = results_dict["accuracy_after_switching"]

    return commonAnalyze(list_of_populations)