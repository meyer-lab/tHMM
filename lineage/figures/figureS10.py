#TODO     
    #tweak state 4
    #add more cells/tweak states for censored cells?

    #Add better visual for best prediction
    #Ideas:
    #add a min marker to each lineage (this works betterfor unnormalized)
    #Histogram plot w/ each graph
    #usually easy to tell for normalized plot so leave it?
    

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
    ax, f = getSetup((10, 6), (2, 4))

    # bern, gamma_a, gamma_scale
    Sone = StateDistribution(0.99, 20, 5)
    Stwo = StateDistribution(0.88, 10, 1)
    Eone = [Sone, Sone]
    Etwo = [Sone, Stwo]
    Ethree = [Sone, Stwo, StateDistribution(0.40, 30, 1)]
    Efour = [Sone, Stwo, StateDistribution(0.40, 30, 1),StateDistribution(0.9, 60, 10)]

    AIC1 = run_AIC(0.02, Eone)
    AIC2 = run_AIC(0.02, Etwo)
    AIC3 = run_AIC(0.02, Ethree)
    AIC4 = run_AIC(0.02, Efour)

    #Finding proper ylim range for all 4 graphs and rounding up
    upper_ylim = int(1+max(np.max(np.ptp(AIC1, axis=0)), np.max(np.ptp(AIC2, axis=0)), np.max(np.ptp(AIC3, axis=0)), np.max(np.ptp(AIC4, axis=0)))/25.0)*25

    AIC5 = run_AIC(0.02, Eone, 10,True)
    AIC6 = run_AIC(0.02, Etwo,10, True)
    AIC7 = run_AIC(0.02, Ethree, 10,True)
    AIC8 = run_AIC(0.02, Efour, 10,True)
    
    #Finding proper ylim range for all 4 censored graphs and rounding up
    upper_ylim_censored= int(1+max(np.max(np.ptp(AIC5, axis=0)), np.max(np.ptp(AIC6, axis=0)), np.max(np.ptp(AIC7, axis=0)), np.max(np.ptp(AIC8, axis=0)))/25.0)*25

    figure_maker(ax[0], AIC1,1,upper_ylim)  
    figure_maker(ax[1], AIC2,2, upper_ylim)
    figure_maker(ax[2], AIC3,3, upper_ylim)
    figure_maker(ax[3], AIC4,4, upper_ylim)
    figure_maker(ax[4], AIC5,1,upper_ylim_censored, True)
    figure_maker(ax[5], AIC6,2, upper_ylim_censored, True)
    figure_maker(ax[6], AIC7,3, upper_ylim_censored, True)
    figure_maker(ax[7], AIC8,4, upper_ylim_censored, True)

    return f




def run_AIC(relative_state_change, E, num_lineages_to_evaluate=10, censored= False):
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
    if censored:
        lineages = [LineageTree.init_from_parameters(pi, T, E, 2**6-1, censor_condition = 3, experiment_time = 1200) for _ in range(num_lineages_to_evaluate)]

    else:
        lineages =  [LineageTree.init_from_parameters(pi, T, E, 2**6-1) for _ in range(num_lineages_to_evaluate)]

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

def figure_maker(ax, AIC_holder, true_state_no, upper_ylim, censored = False):
    """
    Makes figure 10.
    """
    AIC_holder = AIC_holder  - np.min(AIC_holder, axis=0)[np.newaxis, :]
    ax2 = ax.twinx()
    
    ax2.set_ylabel("Number of Lineages Predicted")
    ax2.hist(np.argmin(AIC_holder, axis=0)+1, rwidth = 1, alpha = .2, bins = desired_num_states, align = 'left')
    ax.set_xlabel("Number of States Predicted")
    ax.plot(desired_num_states, AIC_holder, "k", alpha=0.5)
    #ax.plot(np.argmin(AIC_holder, axis = 0)+1, np.min(AIC_holder,axis=0), 'ro', alpha = .5)
    ax.set_ylabel("Normalized AIC")
    ax.margins(0)
    ax2.margins(0)
    ax2.set_yticks(np.linspace(0,10,6))
    ax.set_yticks(np.linspace(0,upper_ylim,len(ax2.get_yticks())))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    title = "Censored " if censored else ""
    title +=f"AIC Under {true_state_no} True "
    title += "States" if true_state_no!=1 else "State"
    ax.set_title(title)
