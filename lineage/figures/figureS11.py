"""
File: figureS11.py
Purpose: Generates figure S11.

AIC.
"""
import numpy as np
from matplotlib.ticker import MaxNLocator

from ..Analyze import run_Analyze_AIC
from ..LineageTree import LineageTree

# States to evaluate with the model
from ..states.StateDistributionGaPhs import StateDistribution

from .figureCommon import getSetup, lineage_good_to_analyze, subplotLabel

# TODO
# Read about/properly derive AICc for this model and check censored lineages with corrected AIC

desired_num_states = np.arange(1, 8)


def makeFigure():
    """
    Makes figure 10.
    """
    ax, f = getSetup((12, 6), (2, 4))

    # Setting up state distributions and E
    Sone = StateDistribution(0.99, 0.9, 10, 2, 10, 2)
    Stwo = StateDistribution(0.9, 0.9, 20, 3, 20, 3)
    Sthree = StateDistribution(0.85, 0.9, 30, 4, 30, 4)
    Sfour = StateDistribution(0.8, 0.9, 40, 5, 40, 5)
    Eone = [Sone, Sone]
    Etwo = [Sone, Stwo]
    Ethree = [Sone, Stwo, Sthree]
    Efour = [Sone, Stwo, Sthree, Sfour]
    E = [Eone, Etwo, Ethree, Efour, Eone, Etwo, Ethree, Efour]

    # making lineages and finding AICs (assign number of lineages here)
    AIC = [run_AIC(.1, e, 10, idx > 3) for idx, e in enumerate(E)]

    # Finding proper ylim range for all 4 uncensored graphs and rounding up
    upper_ylim_uncensored = int(1 + max(np.max(np.ptp(AIC[0], axis=0)), np.max(np.ptp(
        AIC[1], axis=0)), np.max(np.ptp(AIC[2], axis=0)), np.max(np.ptp(AIC[3], axis=0))) / 25.0) * 25

    # Finding proper ylim range for all 4 censored graphs and rounding up
    upper_ylim_censored = int(1 + max(np.max(np.ptp(AIC[4], axis=0)), np.max(np.ptp(
        AIC[5], axis=0)), np.max(np.ptp(AIC[6], axis=0)), np.max(np.ptp(AIC[7], axis=0))) / 25.0) * 25

    upper_ylim = [upper_ylim_uncensored, upper_ylim_censored]

    # Plotting AICs
    for idx, a in enumerate(AIC):
        figure_maker(ax[idx], a, (idx % 4) + 1,
                     upper_ylim[int(idx / 4)], idx > 3)
    subplotLabel(ax)
    return f


def run_AIC(relative_state_change, E, num_lineages_to_evaluate=10, censored=False):
    """
    Run's AIC for known lineages with known pi,
    and T values and stores the output for
    figure drawing.
    """
    # Setting up pi and Transition matrix T:
    #   pi: All states have equal initial probabilities
    #   T:  States have high likelihood of NOT changing, with frequency of change determined mostly by the relative_state_change variable
    #   (as relative state change -> inf state change probabilities approach equality)
    pi = np.ones(len(E)) / len(E)
    T = (np.eye(len(E)) + relative_state_change)
    T = T / np.sum(T, axis=1)[:, np.newaxis]

    # Creating censored lineages
    if censored:
        lineages = [LineageTree.init_from_parameters(
            pi, T, E, 2**6 - 1, censor_condition=3, experiment_time=1200) for _ in range(num_lineages_to_evaluate)]

    # Creating uncensored lineages
    else:
        lineages = [LineageTree.init_from_parameters(
            pi, T, E, 2**6 - 1) for _ in range(num_lineages_to_evaluate)]
    lineages = [l for l in lineages if lineage_good_to_analyze(l)]
    # Storing AICs into array
    AICs = np.empty((len(desired_num_states), len(lineages)))
    output = run_Analyze_AIC(lineages, desired_num_states)
    for idx in range(len(desired_num_states)):
        AIC, _ = output[idx][0].get_AIC(output[idx][2])
        AICs[idx] = np.array([ind_AIC for ind_AIC in AIC])

    return AICs


def figure_maker(ax, AIC_holder, true_state_no, upper_ylim, censored=False):
    """
    Makes figure 10.
    """
    # Normalizing AIC
    AIC_holder = AIC_holder - np.min(AIC_holder, axis=0)[np.newaxis, :]

    # Creating Histogram and setting ylim
    ax2 = ax.twinx()
    ax2.set_ylabel("Lineages Predicted")
    ax2.hist(np.argmin(AIC_holder, axis=0) + 1, rwidth=1,
             alpha=.2, bins=desired_num_states, align='left')
    ax2.set_yticks(np.linspace(0, len(AIC_holder[0]), 1 + len(AIC_holder[0])))

    # Creating AIC plot and matching gridlines
    ax.set_xlabel("Number of States Predicted")
    ax.plot(desired_num_states, AIC_holder, "k", alpha=0.5)
    ax.set_ylabel("Normalized AIC")
    ax.set_yticks(np.linspace(0, upper_ylim, len(ax2.get_yticks())))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Adding title
    title = "Censored " if censored else ""
    title += f"AIC Under {true_state_no} True "
    title += "States" if true_state_no != 1 else "State"
    ax.set_title(title)
