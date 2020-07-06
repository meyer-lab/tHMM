#TODO     
    #look at figure 5
    #init_from_parameters(censor_condition=3, experiment_time = 1200)
    #add a 4th column where there are 4 true states
    #add a min marker to each lineage

"""
File: figure10.py
Purpose: Generates figure 10.

AIC.
"""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from matplotlib.ticker import MaxNLocator

from .figureCommon import getSetup
from ..Analyze import run_Analyze_AIC
from ..LineageTree import LineageTree
from ..states.StateDistributionGamma import StateDistribution
# States to evaluate with the model
desired_num_states = np.arange(1, 8)

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

    AIC1 = run_AIC(0.02, Eone)
    AIC2 = run_AIC(0.02, Etwo)
    AIC3 = run_AIC(0.02, Ethree)
    #Finding proper ylim range for all 3 graphs and rounding up
    upper_ylim = int(1+max(np.max(np.ptp(AIC1, axis=0)), np.max(np.ptp(AIC2, axis=0)), np.max(np.ptp(AIC3, axis=0)))/25.0)*25

    figure_maker(ax[0], AIC1,1,upper_ylim)
    figure_maker(ax[1], AIC2,2, upper_ylim)
    figure_maker(ax[2], AIC3,3, upper_ylim)

    return f




def run_AIC(relative_state_change, E, num_lineages_to_evaluate=10):
    """
    Run's AIC for known lineages with known pi,
    and T values and stores the output for
    figure drawing.
    """
    #Setting up pi and Transition matrix T:
    #   pi: All states have equal initial probabilities
    #   T:  States have high likelihood of NOT changing, with frequency of change determined mostly by the relative_state_change variable
    #           (If relative_state_change>1 then states are more likely to change than stay the same)
    pi = np.ones(len(E))/len(E)
    T = (np.eye(len(E)) + relative_state_change)
    T = T/np.sum(T, axis=1)[:,np.newaxis]

    #Creating lineages from provided E and generated pi and T
    lineages = [LineageTree.init_from_parameters(pi, T, E, 2**6-1) for _ in range(num_lineages_to_evaluate)]

    #Creating np array to store AICs (better for plotting)
    AICs = np.empty((len(desired_num_states), len(lineages)))
    #runnning analysis
    output = run_Analyze_AIC(lineages, desired_num_states)
    #storing AICs from output
    for idx in range(len(desired_num_states)):
        #getting AICs for each lineage from created model
        AIC, _ = output[idx][0].get_AIC(output[idx][2])
        AICs[idx] = np.array([ind_AIC for ind_AIC in AIC])
    
    return AICs 

def figure_maker(ax, AIC_holder, true_state_no, upper_ylim):
    """
    Makes figure 10.
    """
    AIC_holder = AIC_holder  - np.min(AIC_holder, axis=0)[np.newaxis, :]
    ax.set_xlabel("Number of States Predicted")
    ax.plot(desired_num_states, AIC_holder, "k", alpha=0.5)
    ax.set_ylabel("Normalized AIC")
    ax.set_ylim(0.0, upper_ylim)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    title =f"AIC Under {true_state_no} True "
    title += "States" if true_state_no!=1 else "State"
    ax.set_title(title)
