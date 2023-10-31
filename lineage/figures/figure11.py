""" This file depicts the distribution of phase lengths versus the states for each concentration of lapatinib. """

from string import ascii_lowercase
from ..plotTree import plot_networkx, plot_lineage_samples
from ..Analyze import Analyze_list
from ..Lineage_collections import AllLapatinib
from .common import getSetup, plot_all

concs = ["Control", "Lapatinib 25 nM", "Lapatinib 50 nM", "Lapatinib 250 nM"]
concsValues = ["Control", "25 nM", "50 nM", "250 nM"]

num_states = 4
lapt_tHMMobj_list = Analyze_list(AllLapatinib, num_states)[0]

lapt_states_list = [tHMMobj.predict() for tHMMobj in lapt_tHMMobj_list]

# assign the predicted states to each cell
for idx, lapt_tHMMobj in enumerate(lapt_tHMMobj_list):
    for lin_indx, lin in enumerate(lapt_tHMMobj.X):
        for cell_indx, cell in enumerate(lin.output_lineage):
            cell.state = lapt_states_list[idx][lin_indx][cell_indx]

T_lap = lapt_tHMMobj_list[0].estimate.T
num_states = lapt_tHMMobj_list[0].num_states

# plot transition block
plot_networkx(T_lap, "lapatinib")

# plot the sample of lineage trees
plot_lineage_samples(lapt_tHMMobj_list, "figure01")


def makeFigure():
    """Makes figure 11."""

    ax, f = getSetup((17, 7.5), (2, 7))
    plot_all(ax, num_states, lapt_tHMMobj_list, "Laptinib", concs, concsValues)
    for i in range(3, 7):
        ax[i].set_title(concs[i - 3], fontsize=16)
        ax[i].text(
            -0.2,
            1.25,
            ascii_lowercase[i - 2],
            transform=ax[i].transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
        )
        ax[i].axis("off")

    return f
