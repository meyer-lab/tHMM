"""
This creates Figure 10.
"""
from .figureCommon import subplotLabel, getSetup
from matplotlib.ticker import MaxNLocator
from ..Analyze import accuracy, accuracyG, Analyze, getAIC
from ..LineageTree import LineageTree
from ..StateDistribution import StateDistribution
from ..StateDistribution2 import StateDistribution2

import numpy as np
import copy as cp
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


def makeFigure():
    """ Main figure generating function for Fig. 10 """
    ax, f = getSetup((15, 10), (2, 3))

    desred_num_states1, AIC_unpruned1, AIC_pruned1 = AIC_increased_cells1()
    i=0
    figure_maker(ax, i, desred_num_states1, AIC_unpruned1, AIC_pruned1)
    
    desred_num_states2, AIC_unpruned2, AIC_pruned2 = AIC_increased_cells2()
    i=1
    figure_maker(ax, i, desred_num_states2, AIC_unpruned2, AIC_pruned2)
    
    desred_num_states3, AIC_unpruned3, AIC_pruned3 = AIC_increased_cells3()
    i=2
    figure_maker(ax, i, desred_num_states3, AIC_unpruned3, AIC_pruned3)

    return f


def AIC_increased_cells1():
    """ Calculates accuracy and parameter estimation by increasing the number of cells in a lineage for a two-state model. """

    # pi: the initial probability vector
    pi = np.array([1.0, 0.0], dtype="float")

    # T: transition probability matrix
    T = np.array([[1.0, 0.0],
                  [0.0, 1.0]], dtype="float")

    # State 0 parameters "Resistant"
    state0 = 0
    bern_p0 = 0.99
    gamma_a0 = 20
    gamma_loc = 0
    gamma_scale0 = 5

    # State 1 parameters "Susceptible"
    state1 = 1
    bern_p1 = 0.88
    gamma_a1 = 10
    gamma_scale1 = 1

    state_obj0 = StateDistribution(state0, bern_p0, gamma_a0, gamma_loc, gamma_scale0)
    state_obj1 = StateDistribution(state1, bern_p1, gamma_a1, gamma_loc, gamma_scale1)
    E = [state_obj0, state_obj1]
    
    
    desred_num_states = [1, 2, 3]
    times = np.linspace(100, 1000, 10)
    desired_num_cells = 2**11 - 1
    

    AIC_unpruned =  np.zeros(shape=(len(times),len(desred_num_states)))

    for idx, experiment_time in enumerate(times):
        for num_states in desred_num_states:
            # Creating an unpruned and pruned lineage
            lineage_unpruned = LineageTree(pi, T, E, desired_num_cells, experiment_time, prune_condition='both', prune_boolean=True)       

            # Setting then into a list or a population of lineages and collecting the length of each lineage
            X1 = [lineage_unpruned]
            # Analyzing the lineages
            deltas, _, all_states, tHMMobj, _, LL = Analyze(X1, num_states)
            
            # AIC
            AIC_ls, LL_ls, AIC_degrees_of_freedom = getAIC(tHMMobj, LL)
            AIC_unpruned[idx, num_states-1] = (np.mean(AIC_ls))

    return desred_num_states, AIC_unpruned

def AIC_increased_cells2():
    """ Calculates accuracy and parameter estimation by increasing the number of cells in a lineage for a two-state model. """

    # pi: the initial probability vector
    pi = np.array([0.6, 0.4], dtype="float")

    # T: transition probability matrix
    T = np.array([[0.85, 0.15],
                  [0.15, 0.85]], dtype="float")

    # State 0 parameters "Resistant"
    state0 = 0
    bern_p0 = 0.99
    gamma_a0 = 20
    gamma_loc = 0
    gamma_scale0 = 5

    # State 1 parameters "Susceptible"
    state1 = 1
    bern_p1 = 0.91
    gamma_a1 = 10
    gamma_scale1 = 1

    state_obj0 = StateDistribution(state0, bern_p0, gamma_a0, gamma_loc, gamma_scale0)
    state_obj1 = StateDistribution(state1, bern_p1, gamma_a1, gamma_loc, gamma_scale1)
    E = [state_obj0, state_obj1]
    
    
    desred_num_states = [1, 2, 3]
    times = np.linspace(100, 1000, 10)
    desired_num_cells = 2**11 - 1
    

    AIC_unpruned =  np.zeros(shape=(len(times),len(desred_num_states)))

    for idx, experiment_time in enumerate(times):
        for num_states in desred_num_states:
            # Creating an unpruned and pruned lineage
            lineage_unpruned = LineageTree(pi, T, E, desired_num_cells, experiment_time, prune_condition='both', prune_boolean=True)       

            # Setting then into a list or a population of lineages and collecting the length of each lineage
            X1 = [lineage_unpruned]
            # Analyzing the lineages
            deltas, _, all_states, tHMMobj, _, LL = Analyze(X1, num_states)
            
            # AIC
            AIC_ls, LL_ls, AIC_degrees_of_freedom = getAIC(tHMMobj, LL)
            AIC_unpruned[idx, num_states-1] = (np.mean(AIC_ls))

    return desred_num_states, AIC_unpruned


def AIC_increased_cells3():
    """ Calculates accuracy and parameter estimation by increasing the number of cells in a lineage for a two-state model. """

    # pi: the initial probability vector
    pi = np.array([0.5, 0.25, 0.25])

    # T: transition probability matrix
    T = np.array([[0.65, 0.35, 0.00],
                    [0.20, 0.40, 0.40],
                    [0.00, 0.10, 0.90]])

    # E: states are defined as StateDistribution objects

    # State 0 parameters "Susciptible"
    state0 = 0
    bern_p0 = 0.7
    gamma_loc=0
    gamma_a0 = 5.0
    gamma_scale0 = 1.0

    # State 1 parameters "Middle state"
    state1 = 1
    bern_p1 = 0.85
    gamma_a1 = 10.0
    gamma_scale1 = 2.0

    # State 2 parameters "Resistant"
    state2 = 2
    bern_p2 = 0.99
    gamma_a2 = 15.0
    gamma_scale2 = 3.0

    state_obj0 = StateDistribution(state0, bern_p0, gamma_a0, gamma_loc, gamma_scale0)
    state_obj1 = StateDistribution(state1, bern_p1, gamma_a1, gamma_loc, gamma_scale1)
    state_obj2 = StateDistribution(state2, bern_p2, gamma_a2, gamma_loc, gamma_scale2)

    E = [state_obj0, state_obj1, state_obj2]
    
    desred_num_states = [1, 2, 3]
    times = np.linspace(100, 1000, 10)
    desired_num_cells = 2**11 - 1
    

    AIC_unpruned =  np.zeros(shape=(len(times),len(desred_num_states)))

    for idx, experiment_time in enumerate(times):
        for num_states in desred_num_states:
            # Creating an unpruned and pruned lineage
            lineage_unpruned = LineageTree(pi, T, E, desired_num_cells, experiment_time, prune_condition='both', prune_boolean=True)       

            # Setting then into a list or a population of lineages and collecting the length of each lineage
            X1 = [lineage_unpruned]
            # Analyzing the lineages
            deltas, _, all_states, tHMMobj, _, LL = Analyze(X1, num_states)
            
            # AIC
            AIC_ls, LL_ls, AIC_degrees_of_freedom = getAIC(tHMMobj, LL)
            AIC_unpruned[idx, num_states-1] = (np.mean(AIC_ls))

    return desred_num_states, AIC_unpruned

def figure_maker(ax, i, desred_num_states, AIC_unpruned):

    font = 11
    font2 = 10
    i+=0
    ax[i].set_xlim((0, int(np.ceil(1.1 * max(desred_num_states)))))
    ax[i].set_xlabel('Number of States', fontsize=font2)
    ax[i].boxplot(AIC_unpruned)
    ax[i].set_ylabel(r'AIC', rotation=90, fontsize=font2)
    ax[i].get_yticks()
    ax[i].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.25)
    ax[i].set_title('State Assignment AIC', fontsize=font)