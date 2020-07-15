"""
Contains utilities, functions, and variables that are commonly used or shared amongst
the figure creation files.
"""
from string import ascii_lowercase
from cycler import cycler
import numpy as np
from matplotlib import gridspec, pyplot as plt
import seaborn as sns
import svgutils.transform as st
from ..Analyze import run_Results_over, run_Analyze_over

from ..states.StateDistributionGamma import StateDistribution
from ..states.StateDistributionExpon import StateDistribution as expon_state
from ..states.StateDistributionGaPhs import StateDistribution as phaseStateDist

# pi: the initial probability vector
pi = np.array([0.75, 0.25], dtype="float")

# T: transition probability matrix
T = np.array([[0.9, 0.1], [0.3, 0.7]], dtype="float")

# bern, gamma_a, gamma_scale
state0 = StateDistribution(0.99, 8, 6)
state1 = StateDistribution(0.75, 8, 1)
E = [state0, state1]

# bern, exp_beta
state10 = expon_state(0.99, 48)
state11 = expon_state(0.75, 8)
E1 = [state10, state11]

state20 = phaseStateDist(0.99, 0.9, 8, 7, 4, 2)
state21 = phaseStateDist(0.9, 0.85, 6, 4, 3, 5)
E2 = [state20, state21]

min_desired_num_cells = (2 ** 4) - 1
max_desired_num_cells = (2 ** 9) - 1

min_experiment_time = 72
max_experiment_time = 144

min_num_lineages = 1
max_num_lineages = 50

num_data_points = 400


def lineage_good_to_analyze(tmp_lineage, min_lineage_length=10):
    """
    Boolean function that returns True when a lineage is
    good for analysis.
    A lineage is good for analysis when it is heterogeneous,
    that is, contains more than one state in its data, and if
    it is of sufficient length.
    """
    is_sufficient_length = len(tmp_lineage) >= min_lineage_length
    is_heterogeneous = tmp_lineage.is_heterogeneous()
    return is_sufficient_length and is_heterogeneous


def getSetup(figsize, gridd):
    """
    Establish figure set-up with subplots.
    """
    with sns.plotting_context("paper"):
        sns.set(
            palette="deep",
            rc={"axes.facecolor": "#ffffff",  # axes background color
                "axes.edgecolor": "#000000",  # axes edge color
                "axes.xmargin": 0,            # x margin.  See `axes.Axes.margins`
                "axes.ymargin": 0,            # y margin See `axes.Axes.margins`
                "axes.linewidth": 1. / 4,
                "grid.linestyle": "-",
                "grid.alpha": 1. / 4,
                "grid.color": "#000000",
                "xtick.bottom": True,
                "xtick.direction": "inout",
                "xtick.major.width": 1. / 4,  # major tick width in points
                "xtick.minor.width": 0.5 / 4,  # minor tick width in points
                "ytick.left": True,
                "ytick.direction": "inout",
                "ytick.major.width": 1. / 4,  # major tick width in points
                "ytick.minor.width": 0.5 / 4,  # minor tick width in points
                },
        )

        # Setup plotting space and grid
        f = plt.figure(figsize=figsize, constrained_layout=True)
        gs1 = gridspec.GridSpec(*gridd, figure=f)

        # Get list of axis objects
        ax = list()
        for x in range(gridd[0] * gridd[1]):
            ax.append(f.add_subplot(gs1[x]))

    return ax, f


def subplotLabel(axs):
    """
    Sublot labels
    """
    i = 0
    for _, ax in enumerate(axs):
        if ax.has_data() or i == 0:  # only label plots with graphs on them
            ax.text(-0.2, 1.25, ascii_lowercase[i], transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")
            i += 1


def overlayCartoon(figFile, cartoonFile, x, y, scalee=1, scale_x=1, scale_y=1):
    """ Add cartoon to a figure file. """

    # Overlay Figure cartoons
    template = st.fromfile(figFile)
    cartoon = st.fromfile(cartoonFile).getroot()

    cartoon.moveto(x, y, scale=scalee)
    cartoon.scale_xy(scale_x, scale_y)

    template.append(cartoon)
    template.save(figFile)


def commonAnalyze(list_of_populations, xtype="length", **kwargs):
    """
    The standard way of analyzing a list of populations (a list of list of lineages)
    for analysis and plotting.
    """
    list_of_fpi = kwargs.get("list_of_fpi", [None] * len(list_of_populations))
    list_of_fT = kwargs.get("list_of_fT", [None] * len(list_of_populations))
    list_of_fE = kwargs.get("list_of_fE", [None] * len(list_of_populations))
    predicted_num_states = kwargs.get("predicted_num_states", 2)
    parallel = kwargs.get("parallel", True)
    # Analyzing the lineages in the list of populations (parallelized function)
    output = run_Analyze_over(list_of_populations, predicted_num_states, parallel=parallel,
                              list_of_fpi=list_of_fpi, list_of_fT=list_of_fT, list_of_fE=list_of_fE)

    # Collecting the results of analyzing the lineages
    results_holder = run_Results_over(output)

    dictOut = {}

    for key in results_holder[0].keys():
        dictOut[key] = []

    for results_dict in results_holder:
        for key, val in results_dict.items():
            dictOut[key].append(val)

    paramEst = np.array(dictOut["param_estimates"])
    paramTrues = np.array(dictOut["param_trues"])

    x = None
    if xtype == "length":
        x = dictOut["total_number_of_cells"]
    elif xtype == "prop":
        x = dictOut["state_proportions_0"]
    elif xtype == "wass":
        x = dictOut["wasserstein"]
    elif xtype == "bern":
        x = paramTrues[:, 0, 0]

    return x, paramEst, dictOut["balanced_accuracy_score"], dictOut["transition_matrix_norm"], dictOut["pi_vector_norm"], paramTrues


def figureMaker(ax, x, paramEst, accuracies, tr, pii, paramTrues, xlabel="Number of Cells"):
    """
    Makes the common 6 panel figures displaying parameter estimation across lineages
    of various types and sizes.
    """
    # Checks whether we are plotting exponential results, or gamma results
    number_of_params = paramEst.shape[-1]

    i = 0
    ax[i].axis('off')

    i += 1
    ax[i].axis('off')

    i += 1
    ax[i].axis('off')

    i += 1
    ax[i].set_xlabel(xlabel)
    ax[i].scatter(x, paramEst[:, 0, 0], edgecolors="k", marker="o", alpha=0.5)
    ax[i].scatter(x, paramEst[:, 1, 0], edgecolors="k", marker="o", alpha=0.5)
    ax[i].scatter(x, paramTrues[:, 0, 0], marker="_", alpha=0.5)
    ax[i].scatter(x, paramTrues[:, 1, 0], marker="_", alpha=0.5)
    ax[i].set_ylim(bottom=0, top=1.02)
    if number_of_params == 6:
        ax[i].set_ylabel("G1 Bernoulli $p$")
        ax[i].set_title(r"G1 Bernoulli $p$")
    else:
        ax[i].set_ylabel("Bernoulli $p$")
        ax[i].set_title(r"Bernoulli $p$")

    i += 1
    ax[i].set_xlabel(xlabel)
    if number_of_params == 6:
        ax[i].scatter(x, paramEst[:, 0, 2], edgecolors="k",
                      marker="o", alpha=0.5)
        ax[i].scatter(x, paramEst[:, 1, 2], edgecolors="k",
                      marker="o", alpha=0.5)
        ax[i].scatter(x, paramTrues[:, 0, 2], marker="_", alpha=0.5)
        ax[i].scatter(x, paramTrues[:, 1, 2], marker="_", alpha=0.5)
    else:
        ax[i].scatter(x, paramEst[:, 0, 1], edgecolors="k",
                      marker="o", alpha=0.5)
        ax[i].scatter(x, paramEst[:, 1, 1], edgecolors="k",
                      marker="o", alpha=0.5)
        ax[i].scatter(x, paramTrues[:, 0, 1], marker="_", alpha=0.5)
        ax[i].scatter(x, paramTrues[:, 1, 1], marker="_", alpha=0.5)
    if number_of_params == 2:
        ax[i].set_ylabel(r"exponential $\beta$")
        ax[i].set_title(r"exponential $\beta$")
    elif number_of_params == 6:
        ax[i].set_ylabel(r"G1 Gamma $k$")
        ax[i].set_title(r"G1 Gamma $k$")
    else:
        ax[i].set_ylabel(r"Gamma $k$")
        ax[i].set_title(r"Gamma $k$")

    i += 1
    if number_of_params == 2:
        ax[i].axis('off')
    else:
        ax[i].set_xlabel(xlabel)
        if number_of_params == 6:
            ax[i].scatter(x, paramEst[:, 0, 3], edgecolors="k",
                          marker="o", alpha=0.5)
            ax[i].scatter(x, paramEst[:, 1, 3], edgecolors="k",
                          marker="o", alpha=0.5)
            ax[i].scatter(x, paramTrues[:, 0, 3], marker="_",
                          alpha=0.5, label="State 1")
            ax[i].scatter(x, paramTrues[:, 1, 3], marker="_",
                          alpha=0.5, label="State 2")
            ax[i].set_ylabel(r"G1 Gamma $\theta$")
            ax[i].set_title(r"G1 Gamma $\theta$")
        else:
            ax[i].set_xlabel(xlabel)
            ax[i].scatter(x, paramEst[:, 0, 2], edgecolors="k",
                          marker="o", alpha=0.5)
            ax[i].scatter(x, paramEst[:, 1, 2], edgecolors="k",
                          marker="o", alpha=0.5)
            ax[i].scatter(x, paramTrues[:, 0, 2], marker="_",
                          alpha=0.5, label="State 1")
            ax[i].scatter(x, paramTrues[:, 1, 2], marker="_",
                          alpha=0.5, label="State 2")
            ax[i].set_ylabel(r"Gamma $\theta$")
            ax[i].set_title(r"Gamma $\theta$")
        ax[i].legend()

    i += 1
    ax[i].set_xlabel(xlabel)
    if number_of_params == 6:
        ax[i].scatter(x, paramEst[:, 0, 1], edgecolors="k",
                      marker="o", alpha=0.5)
        ax[i].scatter(x, paramEst[:, 1, 1], edgecolors="k",
                      marker="o", alpha=0.5)
        ax[i].scatter(x, paramTrues[:, 0, 1], marker="_", alpha=0.5)
        ax[i].scatter(x, paramTrues[:, 1, 1], marker="_", alpha=0.5)
        ax[i].set_ylim(bottom=0, top=1.02)
        ax[i].set_ylabel("G2 Bernoulli $p$")
        ax[i].set_title(r"G2 Bernoulli $p$")
    else:
        ax[i].set_ylim(bottom=0, top=101)
        ax[i].scatter(x, accuracies, c="k", marker="o",
                      label="Accuracy", edgecolors="k", alpha=0.25)
        ax[i].set_ylabel(r"Accuracy [\%]")
        ax[i].axhline(y=100, linestyle="--", linewidth=2, color="k", alpha=1)
        ax[i].set_title("State Assignment Accuracy")

    i += 1
    ax[i].set_xlabel(xlabel)
    if number_of_params == 6:
        ax[i].scatter(x, paramEst[:, 0, 4], edgecolors="k",
                      marker="o", alpha=0.5)
        ax[i].scatter(x, paramEst[:, 1, 4], edgecolors="k",
                      marker="o", alpha=0.5)
        ax[i].scatter(x, paramTrues[:, 0, 4], marker="_", alpha=0.5)
        ax[i].scatter(x, paramTrues[:, 1, 4], marker="_", alpha=0.5)
        ax[i].set_ylabel(r"G2 Gamma $k$")
        ax[i].set_title(r"G2 Gamma $k$")
    else:
        ax[i].set_ylim(bottom=0, top=max(tr) + 0.2)
        ax[i].scatter(x, tr, c="k", marker="o", edgecolors="k", alpha=0.25)
        ax[i].set_ylabel(r"$||T-T_{est}||_{F}$")
        ax[i].axhline(y=0, linestyle="--", linewidth=2, color="k", alpha=1)
        ax[i].set_title("Transition Matrix Estimation")

    i += 1
    ax[i].set_xlabel(xlabel)
    if number_of_params == 6:
        ax[i].scatter(x, paramEst[:, 0, 5], edgecolors="k",
                      marker="o", alpha=0.5)
        ax[i].scatter(x, paramEst[:, 1, 5], edgecolors="k",
                      marker="o", alpha=0.5)
        ax[i].scatter(x, paramTrues[:, 0, 5], marker="_", alpha=0.5)
        ax[i].scatter(x, paramTrues[:, 1, 5], marker="_", alpha=0.5)
        ax[i].set_ylabel(r"G2 Gamma $\theta$")
        ax[i].set_title(r"G2 Gamma $\theta$")
    else:
        ax[i].set_ylim(bottom=0, top=max(pii) + 0.2)
        ax[i].scatter(x, pii, c="k", marker="o", edgecolors="k", alpha=0.25)
        ax[i].set_ylabel(r"$||\pi-\pi_{est}||_{2}$")
        ax[i].axhline(y=0, linestyle="--", linewidth=2, color="k", alpha=1)
        ax[i].set_title("Initial Probability Matrix Estimation")

    if number_of_params == 6:
        i += 1
        ax[i].set_xlabel(xlabel)
        ax[i].set_ylim(bottom=0, top=101)
        ax[i].scatter(x, accuracies, c="k", marker="o",
                      label="Accuracy", edgecolors="k", alpha=0.25)
        ax[i].set_ylabel(r"Accuracy [\%]")
        ax[i].axhline(y=100, linestyle="--", linewidth=2, color="k", alpha=1)
        ax[i].set_title("State Assignment Accuracy")

        i += 1
        ax[i].set_xlabel(xlabel)
        ax[i].set_ylim(bottom=0, top=max(tr) + 0.2)
        ax[i].scatter(x, tr, c="k", marker="o", edgecolors="k", alpha=0.25)
        ax[i].set_ylabel(r"$||T-T_{est}||_{F}$")
        ax[i].axhline(y=0, linestyle="--", linewidth=2, color="k", alpha=1)
        ax[i].set_title("Transition Matrix Estimation")

        i += 1
        ax[i].set_xlabel(xlabel)
        ax[i].set_ylim(bottom=0, top=max(pii) + 0.2)
        ax[i].scatter(x, pii, c="k", marker="o", edgecolors="k", alpha=0.25)
        ax[i].set_ylabel(r"$||\pi-\pi_{est}||_{2}$")
        ax[i].axhline(y=0, linestyle="--", linewidth=2, color="k", alpha=1)
        ax[i].set_title("Initial Probability Vector Estimation")
