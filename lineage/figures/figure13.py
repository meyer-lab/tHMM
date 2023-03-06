""" Plotting the results for HGF. """
""" This file depicts the distribution of phase lengths versus the states for each concentration of lapatinib. """


from string import ascii_lowercase
from matplotlib import gridspec, rcParams, pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from ..Lineage_collections import GFs
from ..Analyze import Analyze_list
from ..plotTree import plot_networkx, plot_lineage_samples
concs = ["PBS", "EGF", "HGF", "OSM"]

num_states = 3
hgf_tHMMobj_list = Analyze_list(GFs, num_states)[0]

hgf_states_list = [tHMMobj.predict() for tHMMobj in hgf_tHMMobj_list]

# assign the predicted states to each cell
for idx, hgf_tHMMobj in enumerate(hgf_tHMMobj_list):
    for lin_indx, lin in enumerate(hgf_tHMMobj.X):
        for cell_indx, cell in enumerate(lin.output_lineage):
            cell.state = hgf_states_list[idx][lin_indx][cell_indx]

T_hgf = hgf_tHMMobj_list[0].estimate.T

num_states = hgf_tHMMobj_list[0].num_states

rcParams['font.sans-serif'] = "Arial"


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
        for x in range(8):
            ax.append(f.add_subplot(gs1[x]))

        ax.append(f.add_subplot(gs1[1, 2:4]))
        ax.append(f.add_subplot(gs1[1, 4:]))
    return ax, f


# plot transition block
plot_networkx(T_hgf, 'HGF')

# plot sample of lineages
plot_lineage_samples(hgf_tHMMobj_list, 'figure03')


def makeFigure():
    """ Makes figure 91. """

    ax, f = getSetup((16, 7.5), (2, 6))
    plot2(ax, num_states, hgf_tHMMobj_list)
    for i in range(2, 6):
        ax[i].set_title(concs[i - 2], fontsize=16)
        ax[i].text(-0.2, 1.25, ascii_lowercase[i - 1], transform=ax[i].transAxes, fontsize=16, fontweight="bold", va="top")
        ax[i].axis('off')

    return f


def plot1(ax, df1, df2):
    """ helps to avoid duplicating code for plotting the gamma-related emission results and bernoulli. """
    df1[['Growth Factors', 'State1', 'State2', 'State3']].plot(x='Growth Factors', kind='bar', ax=ax[8], color=['lightblue', 'orange', 'green'], rot=0)
    df2[['Growth Factors', 'State1', 'State2', 'State3']].plot(x='Growth Factors', kind='bar', ax=ax[9], color=['lightblue', 'orange', 'green'], rot=0)
    ax[8].set_title("Lifetime")
    ax[8].set_ylabel("Log10-Mean Time [hr]")
    ax[8].set_ylim((0.0, 5.5))
    ax[9].set_title("Fate")
    ax[9].set_ylabel("Division Probability")
    ax[9].set_ylim((0.0, 1.1))

    # legend and xlabel
    for i in range(8, 10):
        ax[i].text(-0.2, 1.25, ascii_lowercase[i - 3], transform=ax[i].transAxes, fontsize=16, fontweight="bold", va="top")


def plot2(ax, num_states, tHMMobj_list):
    for i in range(2):
        ax[i].axis("off")
        ax[6 + i].axis("off")
    ax[0].text(-0.2, 1.25, ascii_lowercase[0], transform=ax[0].transAxes, fontsize=16, fontweight="bold", va="top")

    # lapatinib
    lpt_avg = np.zeros((4, num_states))  # the avg lifetime: num_conc x num_states x num_phases
    bern_lpt = np.zeros((4, num_states))  # bernoulli

    # print parameters and estimated values
    for idx, tHMMobj in enumerate(tHMMobj_list):  # for each concentration data
        for i in range(num_states):
            lpt_avg[idx, i] = np.log10(tHMMobj.estimate.E[i].params[1] * tHMMobj.estimate.E[i].params[2])
            # bernoullis
            bern_lpt[idx, i] = tHMMobj.estimate.E[i].params[0]

    df1 = pd.DataFrame({'Growth Factors': ['PBS', 'EGF', 'HGF', 'OSM'],
                        'State1': lpt_avg[:, 0],
                        'State2': lpt_avg[:, 1],
                        'State3': lpt_avg[:, 2]})

    df2 = pd.DataFrame({'Growth Factors': ['PBS', 'EGF', 'HGF', 'OSM'],
                        'State1': bern_lpt[:, 0],
                        'State2': bern_lpt[:, 1],
                        'State3': bern_lpt[:, 2]
                        })

    plot1(ax, df1, df2)
