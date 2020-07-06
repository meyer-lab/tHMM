"""
File: figure10.py
Purpose: Generates figure 10.

AIC.
"""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from matplotlib.ticker import MaxNLocator

from .figureCommon import getSetup
from ..Analyze import run_Analyze_over, Analyze
from ..LineageTree import LineageTree
from ..states.StateDistributionGamma import StateDistribution


def makeFigure():
    """
    Makes figure 10.
    """
    ax, f = getSetup((7, 3), (1, 3))

    # bern, gamma_a, gamma_scale
    Sone = StateDistribution(0.99, 20, 5)
    Stwo = StateDistribution(0.88, 10, 1)
    Eone = [Sone, Sone]
    Etwo = [Sone, Stwo]
    Ethree = [Sone, Stwo, StateDistribution(0.40, 30, 1)]

    figure_maker(ax[0], run_AIC(0.02, Eone))
    figure_maker(ax[1], run_AIC(0.02, Etwo))
    figure_maker(ax[2], run_AIC(0.02, Ethree))

    return f


# States to evaluate with the model
desired_num_states = np.arange(1, 8)


def run_AIC(relative_state_change, E, num_lineages_to_evaluate=10):
    """
    Run's AIC for known lineages with known pi,
    and T values and stores the output for
    figure drawing.
    """
    pi = np.ones(len(E))/len(E)
    T = (np.eye(len(E)) + relative_state_change)
    T = T/np.sum(T, axis=1)[:,np.newaxis]

    lineages = [LineageTree.init_from_parameters(pi, T, E, 2**6-1) for _ in range(num_lineages_to_evaluate)]

   # AICs = np.empty((len(lineages), num_states_shown)) 
   # for state in range(num_states_shown):
   #     tHMM, _, LL = Analyze(lineages, state+1)
   #     AIC, _ = tHMM.get_AIC(LL)
   #     for lineage in range(len(lineages)):
   #         AICs[lineage][state] = AIC[lineage]
   
    AICs = np.empty((len(lineages), len(desired_num_states)))
    output = run_Analyze_AIC(lineages, desired_num_states)
    for idx, states in enumerate(desired_num_states):
        AIC, _ = output[idx][0].get_AIC(output[idx][2])
        for lineageNo in range(len(lineages)):
            AICs[lineageNo][idx]= AIC[lineageNo]
    
    return AICs.T


def run_Analyze_AIC(population, state_list, **kwargs):
    """
    A function that can be parallelized to speed up figure creation.

    This function is the outermost for-loop we will end up using
    when analyzing heterogenous populations or lineages.

    Analyze is the bottleneck in the figure creation process. The
    rest of the code involved in figure creation deals with collecting
    and computing certain statistics, most of which can be done in an
    additional for loop over the results from Analyze.

    :param population: A list of populations that contain lineages.
    :type: list
    :param state_list: An integer number of states to identify (a hyper-parameter of our model)
    :type state_list: Int
    """
    list_of_fpi = kwargs.get("list_of_fpi", [None] * len(population))
    list_of_fT = kwargs.get("list_of_fT", [None] * len(population))
    list_of_fE = kwargs.get("list_of_fE", [None] * len(population))
    output = []
    exe = ProcessPoolExecutor()

    prom_holder = []
    for idx, num_states in enumerate(state_list):
        prom_holder.append(exe.submit(Analyze, population, num_states, fpi=list_of_fpi[idx], fT=list_of_fT[idx], fE=list_of_fE[idx]))

    for _, prom in enumerate(prom_holder):
        output.append(prom.result())

    return output

def figure_maker(ax, AIC_holder):
    """
    Makes figure 10.
    """
    AIC_holder = AIC_holder  #- np.min(AIC_holder, axis=0)[np.newaxis, :]
    ax.set_xlabel("Number of States")
    ax.plot(desired_num_states, AIC_holder, "k", alpha=0.5)
    ax.set_ylabel("AIC")
    #ax.set_ylim(0.0, 50.0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("State Assignment AIC")
