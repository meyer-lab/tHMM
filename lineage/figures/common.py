"""
Contains utilities, functions, and variables that are commonly used or shared amongst
the figure creation files.
"""
from ..states.StateDistributionGaPhs import StateDistribution as phaseStateDist
from ..states.StateDistributionGamma import StateDistribution
import sys
import time
import matplotlib
from string import ascii_lowercase
import math
import numpy as np
import pandas as pd
from matplotlib import gridspec, rcParams, pyplot as plt
import seaborn as sns
import svgutils.transform as st
from ..Analyze import run_Results_over, run_Analyze_over
from ..BaumWelch import calculate_stationary

matplotlib.use('AGG')


# T: transition probability matrix
T = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=float)

# pi: the initial probability vector
pi = calculate_stationary(T)

# bern, gamma_a, gamma_scale
state0 = StateDistribution(0.99, 96, 0.5)
state1 = StateDistribution(0.75, 80, 0.1)
E = [state0, state1]

state20 = phaseStateDist(0.99, 0.95, 100, 0.5, 80, 0.1)
state21 = phaseStateDist(0.95, 0.9, 75, 0.3, 50, 0.2)
E2 = [state20, state21]

min_desired_num_cells = (2 ** 4) - 1
max_desired_num_cells = (2 ** 7) - 1

min_experiment_time = 72
max_experiment_time = 144

min_num_lineages = 3
max_num_lineages = 40

num_data_points = 100

scatter_state_1_kws = {
    "alpha": 0.5,
    "marker": "+",
    "s": 20,
}

scatter_state_2_kws = {
    "alpha": 0.5,
    "marker": "x",
    "s": 20,
    "color": "green"
}
rcParams['font.sans-serif'] = "Arial"
scatter_kws_list = [scatter_state_1_kws, scatter_state_2_kws]


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
                "svg.fonttype": "none"  # Keep as text
                },
        )

        # Setup plotting space and grid
        f = plt.figure(figsize=figsize, dpi=400, constrained_layout=True)
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


def overlayCartoon(figFile, cartoonFile, x, y, scalee=1, scale_x=1, scale_y=1, rotate=None):
    """ Add cartoon to a figure file. """

    # Overlay Figure cartoons
    template = st.fromfile(figFile)
    cartoon = st.fromfile(cartoonFile).getroot()

    cartoon.moveto(x, y, scale_x=scalee * scale_x, scale_y=scalee * scale_y)
    if rotate:
        cartoon.rotate(rotate, x, y)

    template.append(cartoon)
    template.save(figFile)


def genFigure():
    """ Main figure generation function. """
    fdir = './output/'
    cartoon_dir = r"./lineage/figures/cartoons"
    start = time.time()
    nameOut = 'figure' + sys.argv[1]

    exec('from lineage.figures.' + nameOut + ' import makeFigure', globals())
    ff = makeFigure()
    ff.savefig(fdir + nameOut + '.svg', dpi=ff.dpi,
               bbox_inches='tight', pad_inches=0)

    print(
        f'Figure {sys.argv[1]} is done after {time.time() - start} seconds.\n')

    if sys.argv[1] == 'S01':
        # Overlay Figure 1 cartoon
        overlayCartoon(fdir + 'figureS01.svg',
                       f'{cartoon_dir}/figureS01.svg', 50, 0, scalee=0.78)

    if sys.argv[1] == 'S02':
        # Overlay Figure 2 cartoon
        overlayCartoon(fdir + 'figureS02.svg',
                       f'{cartoon_dir}/figureS02.svg', 80, 0, scalee=0.67)

    if sys.argv[1] == 'S03':
        # Overlay Figure 3 cartoon
        overlayCartoon(fdir + 'figureS03.svg',
                       f'{cartoon_dir}/figureS03.svg', 80, 0, scalee=0.67)

    if sys.argv[1] == 'S04':
        # Overlay Figure 4 cartoon
        overlayCartoon(fdir + 'figureS04.svg',
                       f'{cartoon_dir}/figureS04.svg', 15, 10, scalee=0.54)

    if sys.argv[1] == 'S05':
        # Overlay Figure 5 cartoon
        overlayCartoon(fdir + 'figureS05.svg',
                       f'{cartoon_dir}/figureS05.svg', 15, 10, scalee=0.54)

    if sys.argv[1] == 'S06':
        # Overlay Figure 6 cartoon
        overlayCartoon(fdir + 'figureS06.svg',
                       f'{cartoon_dir}/figureS06.svg', 40, 0, scalee=0.52)

    if sys.argv[1] == 'S07':
        # Overlay Figure 7 cartoon
        overlayCartoon(fdir + 'figureS07.svg',
                       f'{cartoon_dir}/figureS07.svg', 30, 0, scalee=0.52)

    if sys.argv[1] == 'S08':
        # Overlay Figure 8 cartoon
        overlayCartoon(fdir + 'figureS08.svg',
                       f'{cartoon_dir}/figureS08.svg', 110, 0, scalee=0.58)

    if sys.argv[1] == 'S09':
        # Overlay Figure 9 cartoon
        overlayCartoon(fdir + 'figureS09.svg',
                       f'{cartoon_dir}/figureS02.svg', 170, 0, scalee=0.48)

    if sys.argv[1] == 'S10':
        # Overlay Figure 10 cartoon
        overlayCartoon(fdir + 'figureS10.svg',
                       f'{cartoon_dir}/figureS03.svg', 150, 0, scalee=0.47)

    if sys.argv[1] == '1':
        # Overlay Figure 1 cartoon
        overlayCartoon(fdir + 'figure1.svg',
                       f'{cartoon_dir}/xaxis-h.svg', 30, 196, scalee=1.1)
        overlayCartoon(fdir + 'figure1.svg',
                       f'{cartoon_dir}/xaxis-h.svg', 215, 196, scalee=1.1)
        overlayCartoon(fdir + 'figure1.svg',
                       f'{cartoon_dir}/xaxis-h.svg', 395, 196, scalee=1.1)
        overlayCartoon(fdir + 'figure1.svg',
                       f'{cartoon_dir}/fig1ylabel.svg', 15, 75, scalee=0.21)
        overlayCartoon(fdir + 'figure1.svg',
                       f'{cartoon_dir}/fig1ylabel.svg', 195, 75, scalee=0.21)
        overlayCartoon(fdir + 'figure1.svg',
                       f'{cartoon_dir}/fig1ylabel.svg', 375, 75, scalee=0.21)

    if sys.argv[1] == '4':
        # Overlay Figure 4 cartoon
        overlayCartoon(fdir + 'figure4.svg',
                       f'{cartoon_dir}/experimentEndLine.svg', 590, 37, scalee=0.25)

    if sys.argv[1] == '5':
        # Overlay Figure 5 cartoon
        overlayCartoon(fdir + 'figure5.svg',
                       f'{cartoon_dir}/figure5.svg', 5, 5, scalee=1.55)

    if sys.argv[1] == '6':
        # Overlay Figure 6 cartoon
        overlayCartoon(fdir + 'figure6.svg',
                       f'{cartoon_dir}/figure6.svg', 70, 65, scalee=0.6)

    if sys.argv[1] == '11':
        # Overlay Transition block
        overlayCartoon(fdir + 'figure11.svg',
                       f'{cartoon_dir}/figure01.svg', 450, 50, scale_x=0.9, scale_y=1.2)
        # overlayCartoon(fdir + 'figure11.svg',
        #                f'{cartoon_dir}/lapatinib.svg', 5, 80, scalee=0.8)
        overlayCartoon(fdir + 'figure11.svg',
                       f'{cartoon_dir}/xaxis-h.svg', 450, 215, scalee=1.44)
        overlayCartoon(fdir + 'figure11.svg',
                       f'{cartoon_dir}/xaxis-h.svg', 645, 215, scalee=1.44)
        overlayCartoon(fdir + 'figure11.svg',
                       f'{cartoon_dir}/xaxis-h.svg', 840, 215, scalee=1.44)
        overlayCartoon(fdir + 'figure11.svg',
                       f'{cartoon_dir}/xaxis-h.svg', 1040, 215, scalee=1.44)

    if sys.argv[1] == '12':
        # Overlay Transition block
        overlayCartoon(fdir + 'figure12.svg',
                       f'{cartoon_dir}/figure02.svg', 450, 50, scale_x=0.9, scale_y=1.2)
        # overlayCartoon(fdir + 'figure12.svg',
        #                f'{cartoon_dir}/gemcitabine.svg', 10, 70, scalee=1.05)
        overlayCartoon(fdir + 'figure12.svg',
                       f'{cartoon_dir}/xaxis-h.svg', 450, 215, scalee=1.44)
        overlayCartoon(fdir + 'figure12.svg',
                       f'{cartoon_dir}/xaxis-h.svg', 645, 215, scalee=1.44)
        overlayCartoon(fdir + 'figure12.svg',
                       f'{cartoon_dir}/xaxis-h.svg', 840, 215, scalee=1.44)
        overlayCartoon(fdir + 'figure12.svg',
                       f'{cartoon_dir}/xaxis-h.svg', 1040, 215, scalee=1.44)

    if sys.argv[1] == '91':
        # Overlay Transition block
        overlayCartoon(fdir + 'figure91.svg',
                       f'{cartoon_dir}/figure03.svg', 430, 30, scale_x=0.9, scale_y=1.5)
        overlayCartoon(fdir + 'figure91.svg',
                       f'{cartoon_dir}/HGF.svg', 0, 80, scalee=1.1)
        overlayCartoon(fdir + 'figure91.svg',
                       f'{cartoon_dir}/xaxis-h.svg', 430, 260, scalee=1.44)
        overlayCartoon(fdir + 'figure91.svg',
                       f'{cartoon_dir}/xaxis-h.svg', 625, 260, scalee=1.44)
        overlayCartoon(fdir + 'figure91.svg',
                       f'{cartoon_dir}/xaxis-h.svg', 820, 260, scalee=1.4)
        overlayCartoon(fdir + 'figure91.svg',
                       f'{cartoon_dir}/xaxis-h.svg', 1010, 260, scalee=1.3)

    if sys.argv[1] == 'S11':
        overlayCartoon(fdir + 'figureS11.svg',
                       f'{cartoon_dir}/xaxis-h.svg', 40, 1790, scalee=1, scale_x=1.6, scale_y=1.6)
        overlayCartoon(fdir + 'figureS11.svg',
                       f'{cartoon_dir}/xaxis-h.svg', 320, 1790, scalee=1, scale_x=1.6, scale_y=1.6)
        overlayCartoon(fdir + 'figureS11.svg',
                       f'{cartoon_dir}/xaxis-h.svg', 580, 1790, scalee=1, scale_x=1.6, scale_y=1.6)
        overlayCartoon(fdir + 'figureS11.svg',
                       f'{cartoon_dir}/xaxis-h.svg', 850, 1790, scalee=1, scale_x=1.6, scale_y=1.6)

    if sys.argv[1] == 'S12':
        overlayCartoon(fdir + 'figureS12.svg',
                       f'{cartoon_dir}/xaxis-h.svg', 40, 1790, scalee=1, scale_x=1.6, scale_y=1.6)
        overlayCartoon(fdir + 'figureS12.svg',
                       f'{cartoon_dir}/xaxis-h.svg', 320, 1790, scalee=1, scale_x=1.6, scale_y=1.6)
        overlayCartoon(fdir + 'figureS12.svg',
                       f'{cartoon_dir}/xaxis-h.svg', 580, 1790, scalee=1, scale_x=1.6, scale_y=1.6)
        overlayCartoon(fdir + 'figureS12.svg',
                       f'{cartoon_dir}/xaxis-h.svg', 850, 1790, scalee=1, scale_x=1.6, scale_y=1.6)

    if sys.argv[1] == 'S13':
        overlayCartoon(fdir + 'figureS13.svg',
                       f'{cartoon_dir}/figure16.svg', 0, 20, scalee=0.8, scale_y=0.5)
        overlayCartoon(fdir + 'figureS13.svg',
                       f'{cartoon_dir}/figure161.svg', 200, 20, scalee=0.8, scale_y=0.5)

    if sys.argv[1] == 'S14':
        # Overlay Transition block
        overlayCartoon(fdir + 'figureS14.svg',
                       f'{cartoon_dir}/figure17.svg', 0, 20, scalee=0.8, scale_y=0.5)
        overlayCartoon(fdir + 'figureS14.svg',
                       f'{cartoon_dir}/figure171.svg', 200, 20, scalee=0.8, scale_y=0.5)


def commonAnalyze(list_of_populations, num_states, xtype="length", **kwargs):
    """
    The standard way of analyzing a list of populations (a list of list of lineages)
    for analysis and plotting.
    """
    list_of_fpi = kwargs.get("list_of_fpi", [None] * len(list_of_populations))
    list_of_fT = kwargs.get("list_of_fT", [None] * len(list_of_populations))
    list_of_fE = kwargs.get("list_of_fE", [None] * len(list_of_populations))

    parallel = kwargs.get("parallel", True)
    # Analyzing the lineages in the list of populations (parallelized function)
    output = run_Analyze_over(list_of_populations, num_states, parallel=parallel,
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

    return x, paramEst, dictOut, paramTrues


def figureMaker(ax, x, paramEst, dictOut, paramTrues, xlabel="Number of Cells", num_lineages=None, dist_dist=False):
    """
    Makes the common 6 panel figures displaying parameter estimation across lineages
    of various types and sizes.
    """
    # Checks whether we are plotting exponential results, or gamma results
    number_of_params = paramEst.shape[-1]
    accuracies = dictOut["state_similarity"]
    tr = dictOut["transition_matrix_similarity"]
    pii = dictOut["pi_similarity"]

    accuracy_df = pd.DataFrame(columns=["x", 'accuracy'])
    accuracy_df['x'] = x
    accuracy_df['accuracy'] = accuracies
    accuracy_df['tr'] = tr
    accuracy_df['pii'] = pii
    accuracy_df['bern 0 0'] = paramEst[:, 0, 0]  # bern G1 or Bern
    accuracy_df['bern 1 0'] = paramEst[:, 1, 0]
    if number_of_params == 6:
        accuracy_df['bern 0 1'] = paramEst[:, 0, 1]  # bern G2
        accuracy_df['bern 1 1'] = paramEst[:, 1, 1]
        accuracy_df['gamma 0 2'] = paramEst[:, 0, 2]  # gamma G1 shape
        accuracy_df['gamma 1 2'] = paramEst[:, 1, 2]
        accuracy_df['gamma 0 3'] = paramEst[:, 0, 3]  # gamma G1 scale
        accuracy_df['gamma 1 3'] = paramEst[:, 1, 3]
        accuracy_df['gamma 0 4'] = paramEst[:, 0, 4]  # gamma G2 shape
        accuracy_df['gamma 1 4'] = paramEst[:, 1, 4]
        accuracy_df['gamma 0 5'] = paramEst[:, 0, 5]  # gamma G2 scale
        accuracy_df['gamma 1 5'] = paramEst[:, 1, 5]
        accuracy_df['wasserstein distance 0'] = dictOut["distribution distance 0"]
        accuracy_df['wasserstein distance 1'] = dictOut["distribution distance 1"]
    else:
        accuracy_df['0 1'] = paramEst[:, 0, 1]  # gamma shape
        accuracy_df['1 1'] = paramEst[:, 1, 1]
        accuracy_df['gamma 0 2'] = paramEst[:, 0, 2]  # gamma scale
        accuracy_df['gamma 1 2'] = paramEst[:, 1, 2]
    if num_lineages is not None:
        accuracy_df['num lineages'] = num_lineages

    i = 0
    ax[i].axis('off')

    i += 1  # i = 1
    ax[i].axis('off')

    i += 1  # i = 2
    ax[i].axis('off')

    i += 1  # i = 3: plot estimation of bernoulli parameter
    sns.regplot(x="x", y="bern 0 0", data=accuracy_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[0])
    sns.regplot(x="x", y="bern 1 0", data=accuracy_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[1], line_kws={"color": "green"})
    ax[i].axhline(y=paramTrues[1, 0, 0], linestyle="-", c="#00ffff", alpha=0.5)
    ax[i].axhline(y=paramTrues[1, 1, 0], linestyle="-", c="#00cc00", alpha=0.5)
    ax[i].set_xlabel(xlabel)
    ax[i].set_ylim(bottom=0.5, top=1.02)
    if number_of_params == 6:
        ax[i].set_ylabel("G1 Bernoulli p")
        ax[i].set_title(r"G1 Bernoulli p Estimation")
    else:
        ax[i].set_ylabel("Bernoulli p")
        ax[i].set_title(r"Bernoulli p Estimation")

    i += 1  # i = 4
    if number_of_params == 6:  # phase specific gamma
        if dist_dist:  # bernoulli G2
            sns.regplot(x="x", y="bern 0 1", data=accuracy_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[0])
            sns.regplot(x="x", y="bern 1 1", data=accuracy_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[1], line_kws={"color": "green"})
            ax[i].scatter(x, paramTrues[:, 0, 1], marker="_", s=20, c="#00ffff", alpha=0.5)
            ax[i].scatter(x, paramTrues[:, 1, 1], marker="_", s=20, c="#00cc00", alpha=0.5)
            ax[i].set_ylim(bottom=0.5, top=1.02)
            ax[i].set_ylabel("S/G2 Bernoulli p")
            ax[i].set_title(r"S/G2 Bernoulli p Estimation")
        else:
            sns.regplot(x="x", y="gamma 0 2", data=accuracy_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[0])
            sns.regplot(x="x", y="gamma 1 2", data=accuracy_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[1], line_kws={"color": "green"})
            ax[i].axhline(y=paramTrues[1, 0, 2], linestyle="-", c="#00ffff", alpha=0.5)
            ax[i].axhline(y=paramTrues[1, 1, 2], linestyle="-", c="#00cc00", alpha=0.5)
            ax[i].set_ylabel(r"G1 Gamma k")
            ax[i].set_title(r"G1 Gamma k Estimation")
            ax[i].set_ylim([0.0, 15.0])

    else:  # simple lifetime gamma
        sns.regplot(x="x", y="0 1", data=accuracy_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[0])
        sns.regplot(x="x", y="1 1", data=accuracy_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[1], line_kws={"color": "green"})
        ax[i].axhline(y=paramTrues[1, 0, 1], linestyle="-", c="#00ffff", alpha=0.5)
        ax[i].axhline(y=paramTrues[1, 1, 1], linestyle="-", c="#00cc00", alpha=0.5)
        ax[i].set_ylabel(r"Gamma k")
        ax[i].set_title(r"Gamma k Estimation")
        ax[i].set_ylim([0.0, 15.0])
    ax[i].set_xlabel(xlabel)

    i += 1  # i = 5
    if number_of_params == 6:
        if dist_dist:  # plot gamma distance
            sns.regplot(x="x", y='wasserstein distance 0', data=accuracy_df, ax=ax[i], lowess=True, label="state 1", marker='+', scatter_kws=scatter_kws_list[0])
            sns.regplot(x="x", y='wasserstein distance 1', data=accuracy_df, ax=ax[i], lowess=True, label="state 2", marker='+', scatter_kws=scatter_kws_list[1], line_kws={"color": "green"})
            ax[i].set_title(r"Distance bw true and estm. gamma dists")
            ax[i].set_ylabel(r"Wasserstein Distance")
            ax[i].set_xlabel(xlabel)
            ax[i].set_ylim(0.0, 10.0)
            ax[i].legend()
        else:  # plot gamma params scale
            sns.regplot(x="x", y="gamma 0 3", data=accuracy_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[0])
            sns.regplot(x="x", y="gamma 1 3", data=accuracy_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[1], line_kws={"color": "green"})
            ax[i].axhline(y=paramTrues[1, 0, 3], linestyle="-",
                          c="#00ffff", alpha=0.5, label="State 1")
            ax[i].axhline(y=paramTrues[1, 1, 3], linestyle="-",
                          c="#00cc00", alpha=0.5, label="State 2")
            ax[i].set_xlabel(xlabel)
            ax[i].set_ylabel(r"G1 Gamma $\theta$")
            ax[i].set_title(r"G1 Gamma $\theta$ Estimation")
            ax[i].set_ylim([0.0, 15.0])
    else:  # just simple gamma params
        sns.regplot(x="x", y="gamma 0 2", data=accuracy_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[0])
        sns.regplot(x="x", y="gamma 1 2", data=accuracy_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[1], line_kws={"color": "green"})
        ax[i].scatter(x, paramTrues[:, 0, 2], marker="_", s=20,
                      c="#00ffff", alpha=0.5, label="State 1")
        ax[i].scatter(x, paramTrues[:, 1, 2], marker="_", s=20,
                      c="#00cc00", alpha=0.5, label="State 2")
        ax[i].set_xlabel(xlabel)
        ax[i].set_ylabel(r"Gamma $\theta$")
        ax[i].set_title(r"Gamma $\theta$ Estimation")
        ax[i].set_ylim([0.0, 15.0])
    ax[i].legend()

    i += 1  # i = 6
    if number_of_params == 6 and (not dist_dist):
        sns.regplot(x="x", y="bern 0 1", data=accuracy_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[0])
        sns.regplot(x="x", y="bern 1 1", data=accuracy_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[1], line_kws={"color": "green"})
        ax[i].scatter(x, paramTrues[:, 0, 1], marker="_", s=20, c="#00ffff", alpha=0.5)
        ax[i].scatter(x, paramTrues[:, 1, 1], marker="_", s=20, c="#00cc00", alpha=0.5)
        ax[i].set_ylim(bottom=0.0, top=1.02)
        ax[i].set_ylabel("S/G2 Bernoulli p")
        ax[i].set_title(r"S/G2 Bernoulli p")
    else:
        ax[i].set_ylim(bottom=0, top=101)
        sns.regplot(x="x", y="accuracy", data=accuracy_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[0])
        ax[i].set_ylabel(r"Adjusted Rand Index Accuracy [%]")
        ax[i].set_title("State Assignment Accuracy")
    ax[i].set_xlabel(xlabel)

    i += 1  # i = 7
    if number_of_params == 6 and (not dist_dist):
        sns.regplot(x="x", y="gamma 0 4", data=accuracy_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[0])
        sns.regplot(x="x", y="gamma 1 4", data=accuracy_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[1], line_kws={"color": "green"})
        ax[i].scatter(x, paramTrues[:, 0, 4], marker="_", s=20, c="#00ffff", alpha=0.5)
        ax[i].scatter(x, paramTrues[:, 1, 4], marker="_", s=20, c="#00cc00", alpha=0.5)
        ax[i].set_ylabel(r"S/G2 Gamma $k$")
        ax[i].set_title(r"S/G2 Gamma $k$")
        ax[i].set_ylim([0.0, 15.0])
    else:
        ax[i].set_ylim(bottom=0, top=np.mean(tr) + 0.2)
        sns.regplot(x="x", y="tr", data=accuracy_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[0])
        ax[i].set_ylabel(r"$||T-T_{est}||_{F}$")
        ax[i].set_title(r"Error in Estimating T")
        ax[i].set_ylim([0.0, 1.0])
    ax[i].set_xlabel(xlabel)

    i += 1  # i = 8 (last)
    if number_of_params == 6 and (not dist_dist):
        sns.regplot(x="x", y="gamma 0 5", data=accuracy_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[0])
        sns.regplot(x="x", y="gamma 1 5", data=accuracy_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[1], line_kws={"color": "green"})
        ax[i].scatter(x, paramTrues[:, 0, 5], marker="_", s=20, c="#00ffff", alpha=0.5)
        ax[i].scatter(x, paramTrues[:, 1, 5], marker="_", s=20, c="#00cc00", alpha=0.5)
        ax[i].set_ylabel(r"S/G2 Gamma $\theta$")
        ax[i].set_title(r"S/G2 Gamma $\theta$")
        ax[i].set_xlabel(xlabel)
        ax[i].set_ylim([0.0, 15.0])
    else:
        if (len(accuracy_df["pii"].unique()) <= math.factorial(paramTrues.shape[1])) or (num_lineages is None):
            ax[i].axis('off')
        else:
            ax[i].set_ylim(bottom=0, top=np.mean(pii) + 0.2)
            sns.regplot(x="num lineages", y="pii", data=accuracy_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[0])
            ax[i].set_ylabel(r"$||\pi-\pi_{est}||_{2}$")
            ax[i].set_title(r"Error in Estimating $\pi$")
            ax[i].set_xlabel("Number of Lineages")
            ax[i].set_ylim([0.0, 1.0])

    if number_of_params == 6 and (not dist_dist):
        i += 1
        ax[i].set_ylim(bottom=0, top=101)
        ax[i].set_ylabel(r"Adjusted Rand Index Accuracy [%]")
        sns.regplot(x="x", y="accuracy", data=accuracy_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[0])
        ax[i].set_title("State Assignment Accuracy")
        ax[i].set_xlabel(xlabel)

        i += 1
        ax[i].set_ylim(bottom=0, top=np.mean(tr) + 0.2)
        sns.regplot(x="x", y="tr", data=accuracy_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[0])
        ax[i].set_ylabel(r"$||T-T_{est}||_{F}$")
        ax[i].set_title(r"Error in Estimating T")
        ax[i].set_ylim([0.0, 1.0])
        ax[i].set_xlabel(xlabel)

        i += 1
        if (len(accuracy_df["pii"].unique()) <= math.factorial(paramTrues.shape[1])) or (num_lineages is None):
            ax[i].axis('off')
        else:
            ax[i].set_ylim(bottom=0, top=np.mean(pii) + 0.2)
            sns.regplot(x="num lineages", y="pii", data=accuracy_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[0])
            ax[i].set_ylabel(r"$||\pi-\pi_{est}||_{2}$")
            ax[i].set_title(r"Error in Estimating $\pi$")
            ax[i].set_xlabel("Number of Lineages")
            ax[i].set_ylim([0.0, 1.0])


colors = ['blue', 'orange', 'green', 'red', 'purple', 'grey']
def plotting(ax, lpt_avg, bern_lpt, cons, concsValues, num_states):
    """ helps to avoid duplicating code for plotting the gamma-related emission results and bernoulli. """
    for i in range(num_states):  # lapatinib that has 3 states
        ax[10].plot(cons, lpt_avg[:, i, 0], label="state " + str(i + 1), color=colors[i], alpha=0.7)
        ax[10].set_title("G1 phase")
        ax[11].plot(cons, lpt_avg[:, i, 1], label="state " + str(i + 1), color=colors[i], alpha=0.7)
        ax[11].set_title("S/G2 phase")
        ax[12].plot(cons, bern_lpt[:, i, 0], label="state " + str(i + 1), color=colors[i], alpha=0.7)
        ax[12].set_title("G1 phase")
        ax[13].plot(cons, bern_lpt[:, i, 1], label="state " + str(i + 1), color=colors[i], alpha=0.7)
        ax[13].set_title("S/G2 phase")

    # ylim and ylabel
    for i in range(10, 12):
        ax[i].set_ylabel("Mean Time [hr]")
        ax[i].set_ylim([0, 100.0])

    # ylim and ylabel
    for i in range(12, 14):
        ax[i].set_ylabel("Division Probability")
        ax[i].set_ylim([0.0, 1.05])

    # legend and xlabel
    for i in range(10, 14):
        ax[i].legend()
        ax[i].set_xlabel("Concentration [nM]")
        ax[i].set_xticklabels(concsValues, rotation=30)
        ax[i].text(-0.2, 1.25, ascii_lowercase[i - 5], transform=ax[i].transAxes, fontsize=16, fontweight="bold", va="top")


def plot_all(ax, num_states, tHMMobj_list, Dname, cons, concsValues):
    for i in range(3):
        ax[i].axis("off")
        ax[7 + i].axis("off")
    ax[0].text(-0.2, 1.25, ascii_lowercase[0], transform=ax[0].transAxes, fontsize=16, fontweight="bold", va="top")

    # lapatinib
    lpt_avg = np.zeros((4, num_states, 2))  # the avg lifetime: num_conc x num_states x num_phases
    bern_lpt = np.zeros((4, num_states, 2))  # bernoulli
    # print parameters and estimated values
    print(Dname, "\n the \u03C0: ", tHMMobj_list[0].estimate.pi, "\n the transition matrix: ", tHMMobj_list[0].estimate.T)
    for idx, tHMMobj in enumerate(tHMMobj_list):  # for each concentration data
        for i in range(num_states):
            lpt_avg[idx, i, 0] = tHMMobj.estimate.E[i].params[2] * tHMMobj.estimate.E[i].params[3]  # G1
            lpt_avg[idx, i, 1] = tHMMobj.estimate.E[i].params[4] * tHMMobj.estimate.E[i].params[5]  # G2
            # bernoullis
            for j in range(2):
                bern_lpt[idx, i, j] = tHMMobj.estimate.E[i].params[j]

    plotting(ax, lpt_avg, bern_lpt, cons, concsValues, num_states)


def sort_lins(tHMMobj):
    """ Sorts lineages based on their root cell state for plotting the lineage trees. """
    num_st = tHMMobj.estimate.num_states

    st = []  # holds the state of root cell in all lineages for this particular tHMMobj
    for lins in tHMMobj.X:
        st.append(lins.output_lineage[0].state)

    states = []
    for i in range(num_st):
        st_i = [index for index, val in enumerate(st) if val == i]
        temp = [tHMMobj.X[k] for k in st_i[0:15]]

        states += temp

    return states
