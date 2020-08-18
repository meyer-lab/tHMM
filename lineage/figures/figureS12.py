"""
File: figureS12.py
Purpose: Generates figure S12.
Figure S12 shows the accuracy of state assignment versus 
increasing cell numbers for 2-state, 3-state, and 4-state models.
"""
import numpy as np
from ..Analyze import Analyze, Results, run_Analyze_over
from ..states.StateDistributionGaPhs import StateDistribution as StateDistPhase

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
    ax, f = getSetup((9, 3), (1, 3))

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

    pi2 = np.array([0.60, 0.40])
    T2 = np.array([[0.75, 0.25], [0.1, 0.90]])
    E2 = [StateDistPhase(0.99, 0.9, 20, 5, 10, 3), StateDistPhase(0.88, 0.75, 10, 2, 15, 4)]
    
    X_2 = [LineageTree.init_from_parameters(pi2, T2, E2, desired_num_cells=(2 ** 11) - 1)]
    
    pi3 = np.array([0.55, 0.35, 0.10]) 
    T3 = np.array([[0.75, 0.20, 0.05], [0.1, 0.85, 0.05], [0.1, 0.1, 0.8]])
    E3 = [StateDistPhase(0.99, 0.9, 20, 5, 10, 3), StateDistPhase(0.88, 0.75, 10, 2, 15, 4), StateDistPhase(0.77, 0.85, 15, 7, 20, 5)]
    X_3 = [LineageTree.init_from_parameters(pi, T, E, desired_num_cells=(2 ** 11) - 1)]
    
    pi4 = np.array([0.55, 0.35, 0.06, 0.04])
    T4 = np.array([[0.70, 0.20, 0.05, 0.05], [0.1, 0.80, 0.06, 0.04], [0.1, 0.1, 0.6, 0.02], [0.1, 0.1, 0.5, 0.03]])
    E4 = [StateDistPhase(0.99, 0.9, 20, 5, 10, 3), StateDistPhase(0.88, 0.75, 10, 2, 15, 4), StateDistPhase(0.77, 0.85, 15, 7, 20, 5), StateDistPhase(0.66, 0.95, 17, 6, 15, 5)]
    
    X_4 = [LineageTree.init_from_parameters(pi4, T4, E4, desired_num_cells=(2 ** 11) - 1)]
    
    X2 = []
    X3 = []
    X4 = []
    
    #2 state population
    num_lineages = np.linspace(min_num_lineages, int(0.35 * max_num_lineages), num_data_points, dtype=int)
    list_of_populations2 = []
    
    for indx, num in enumerate(num_lineages):
        population2 = []
    
        for _ in range(10):
            good2go = False
            while not good2go:
                tmp_lineage2 = LineageTree.init_from_parameters(pi2, T2, E2, 2 ** 11 - 1)
                good2go = lineage_good_to_analyze(tmp_lineage2)
            
            population2.append(tmp_lineage2)
    
    tree_obj, predicted_states, LL = run_Analyze_over(population2, 2)
    results_dict = Results(tree_obj, predicted_states, LL)
    accuracy_2 = results_dict["accuracy_after_switching"]

    #3 state population
    list_of_populations3 = []
    
    for indx, num in enumerate(num_lineages):
        population3 = []
        for _ in range(10):
            good2go = False
            while not good2go:
                tmp_lineage3 = LineageTree.init_from_parameters(pi3, T3, E3, 2 ** 11 - 1)
                good2go = lineage_good_to_analyze(tmp_lineage3)
            
            population3.append(tmp_lineage3)
    
    tree_obj, predicted_states, LL = run_Analyze_over(population3, 3)
    results_dict = Results(tree_obj, predicted_states, LL)
    accuracy_3 = results_dict["accuracy_after_switching"]
            
    #4 state population
    list_of_populations4 = []
    
    for indx, num in enumerate(num_lineages):
        population4 = []
        for _ in range(10):
            good2go = False
            while not good2go:
                tmp_lineage4 = LineageTree.init_from_parameters(pi4, T4, E4, 2 ** 11 - 1)
                good2go = lineage_good_to_analyze(tmp_lineage4)
            
            population4.append(tmp_lineage4)
    
    tree_obj, predicted_states, LL = run_Analyze_over(population4, 4)
    results_dict = Results(tree_obj, predicted_states, LL)
    accuracy_4 = results_dict["accuracy_after_switching"]

    return accuracy_2, accuracy_3, accuracy_4