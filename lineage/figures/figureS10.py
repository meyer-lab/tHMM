"""
File: figure10.py
Purpose: Generates figure 10.

AIC.
"""
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
    ax, f = getSetup((7, 3), (1, 4))

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
desired_num_states = np.arange(1, 6)


def run_AIC(relative_state_change, E, num_to_evaluate=10):
    """
    Run's AIC for known lineages with known pi,
    and T values and stores the output for
    figure drawing.
    """
    num_states_shown = 5
    pi = np.ones(len(E))/len(E)
    T = (np.eye(len(E)) + relative_state_change)
    T = T/np.sum(T, axis=1)[:,np.newaxis]

    lineages = []
    for _ in range(num_to_evaluate):
        lineages.append([LineageTree.init_from_parameters(pi, T, E, 2**7-1)])
    
    ls = []
    for lineage in lineages:
        ls.append(lineage[0])
    AICs = np.empty((len(lineages), num_states_shown)) 
    for state in range(num_states_shown):
        tHMM, _, LL = Analyze(ls, state+1)
        AIC, _ = tHMM.get_AIC(LL)
        for lineage in range(len(lineages)):
            AICs[lineage][state] = AIC[lineage]
   
   # AICs = np.empty((len(lineages), num_states_shown))
   # for states in range(0,num_states_shown):
       # output = run_Analyze_over(lineages, states+1)
       # for lineageNo in range(len(lineages)):  
           # AIC, _ = output[lineageNo][0].get_AIC(output[lineageNo][2])
           # AICs[lineageNo][states]= AIC[0]
    
    return AICs.T


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
