"""
File: figureS12.py
Purpose: Generates figure S12.
Figure S12 shows the accuracy of state assignment versus 
increasing cell numbers for 2-state, 3-state, and 4-state models.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from ..states.StateDistributionGaPhs import StateDistribution as StateDistPhase

from .figureCommon import (
    getSetup,
    subplotLabel,
    commonAnalyze,
    pi,
    T,
    E,
    min_desired_num_cells,
    lineage_good_to_analyze,
    min_num_lineages,
    max_num_lineages,
    num_data_points,
    scatter_kws_list,
)
from ..LineageTree import LineageTree


def makeFigure():
    """
    Makes figure S12.
    """

    # Get list of axis objects
    ax, f = getSetup((9, 3), (1, 3))

    acc_df = accuracy()
    figureMaker(ax, acc_df)

    subplotLabel(ax)

    return f


def accuracy():
    """
    Calculates accuracy and parameter estimation
    over an increasing number of lineages in a population for
    a uncensored lineages of 2, 3, and 4 states.
    We increase the desired number of cells in a lineage by
    the experiment time.
    """
        
    pi3 = np.array([0.55, 0.35, 0.10]) 
    T3 = np.array([[0.75, 0.20, 0.05], [0.1, 0.85, 0.05], [0.1, 0.1, 0.8]])
    E3 = [StateDistPhase(0.99, 0.9, 20, 5, 10, 3), StateDistPhase(0.88, 0.75, 10, 2, 15, 4), StateDistPhase(0.77, 0.85, 15, 7, 20, 5)]
    
    pi4 = np.array([0.55, 0.35, 0.06, 0.04])
    T4 = np.array([[0.70, 0.20, 0.05, 0.05], [0.1, 0.80, 0.06, 0.04], [0.1, 0.1, 0.6, 0.2], [0.1, 0.1, 0.5, 0.3]])
    E4 = [StateDistPhase(0.99, 0.9, 20, 5, 10, 3), StateDistPhase(0.88, 0.75, 10, 2, 15, 4), StateDistPhase(0.77, 0.85, 15, 7, 20, 5), StateDistPhase(0.66, 0.95, 17, 6, 15, 5)]

    # common for all three populations
    num_lineages = np.linspace(min_num_lineages, 50, 50, dtype=int)
    experiment_times = np.linspace(1000, int(2.5 * 1000), num_data_points)
    
    #2 state population
    list_of_populations2 = []

    for indx, num in enumerate(num_lineages):
        population2 = []
        for _ in range(num):

            good2go = False
            while not good2go:
                tmp_lineage2 = LineageTree.init_from_parameters(pi, T, E, desired_num_cells=min_desired_num_cells, censor_condition=3, desired_experiment_time=experiment_times[indx])
                good2go = lineage_good_to_analyze(tmp_lineage2)

            population2.append(tmp_lineage2)

        # Adding populations into a holder for analysing
        list_of_populations2.append(population2)
    print ("before state 2 analyze")
    cell_number_x2, _, accuracy2_after_switching, _, _, _ = commonAnalyze(list_of_populations2, 2)
    print ("after state 2 analyze")
    #3 state population
    list_of_populations3 = []
    
    for indx, num in enumerate(num_lineages):
        population3 = []
        for _ in range(num):
            
            good2go = False
            while not good2go:
                tmp_lineage3 = LineageTree.init_from_parameters(pi3, T3, E3, desired_num_cells=min_desired_num_cells, censor_condition=3, desired_experiment_time=experiment_times[indx])
                good2go = lineage_good_to_analyze(tmp_lineage3)
            
            population3.append(tmp_lineage3)
    
    # Adding populations into a holder for analysing
        list_of_populations3.append(population3)
    print ("before state 3 analyze")
    cell_number_x3, _, accuracy3_after_switching, _, _, _ = commonAnalyze(list_of_populations3, 3)
    print ("after state 3 analyze")
    #4 state population
    list_of_populations4 = []
    
    for indx, num in enumerate(num_lineages):
        population4 = []
        for _ in range(10):
            good2go = False
            while not good2go:
                tmp_lineage4 = LineageTree.init_from_parameters(pi4, T4, E4, desired_num_cells=min_desired_num_cells, censor_condition=3, desired_experiment_time=experiment_times[indx])
                good2go = lineage_good_to_analyze(tmp_lineage4)
            
            population4.append(tmp_lineage4)
    
    # Adding populations into a holder for analysing
        list_of_populations4.append(population4)
    print ("before state 4 analyze")
    cell_number_x4, _, accuracy4_after_switching, _, _, _ = commonAnalyze(list_of_populations4, 4)
    print ("after state 4 analyze")
    # Create the dataframe for the data.
    accuracy_df = pd.DataFrame(columns=["x2", "x3", "x4", "accuracy2", "accuracy3", "accuracy4"])
    accuracy_df["x2"] = cell_number_x2
    accuracy_df["accuracy2"] = accuracy2_after_switching
    accuracy_df["x3"] = cell_number_x3
    accuracy_df["accuracy3"] = accuracy3_after_switching
    accuracy_df["x4"] = cell_number_x4
    accuracy_df["accuracy4"] = accuracy4_after_switching
    print ("after accuracy df")

    return accuracy_df

def figureMaker(ax, accuracy_df):
    """ This creates figure S12. Includes 3 subplots for accuracy of state assignment for 2, 3, and 4 states. """

    i = 0
    sns.regplot(x="x2", y="accuracy2", data=accuracy_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[0])
    ax[i].set_title("2 State Assignment Accuracy")
    ax[i].set_ylabel("Accuracy [%]")
    ax[i].set_ylim(bottom=25.0, top=101)

    i += 1
    sns.regplot(x="x3", y="accuracy3", data=accuracy_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[0])
    ax[i].set_title("3 State Assignment Accuracy")
    ax[i].set_ylabel("Accuracy [%]")
    ax[i].set_ylim(bottom=25.0, top=101)
    
    i += 1
    sns.regplot(x="x4", y="accuracy4", data=accuracy_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[0])
    ax[i].set_title("4 State Assignment Accuracy")
    ax[i].set_ylabel("Accuracy [%]")
    ax[i].set_ylim(bottom=25.0, top=101)

