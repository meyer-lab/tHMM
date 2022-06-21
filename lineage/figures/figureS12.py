""" This is a file to put together 4 conditions of gemcitabine together. """
from string import ascii_lowercase
import pickle
from .common import getSetup, sort_lins
from ..plotTree import plotLineage


pik1 = open("gemcitabines.pkl", "rb")
alls = []
for i in range(7):
    gemc_tHMMobj_list = []
    for i in range(4):
        gemc_tHMMobj_list.append(pickle.load(pik1))
    alls.append(gemc_tHMMobj_list)

# selected for gemcitabine is 5 states which is index 4.
gemc_tHMMobj_list = alls[4]

gemc_states_list = [tHMMobj.predict() for tHMMobj in gemc_tHMMobj_list]

for idx, gemc_tHMMobj in enumerate(gemc_tHMMobj_list):
    for lin_indx, lin in enumerate(gemc_tHMMobj.X):
        for cell_indx, cell in enumerate(lin.output_lineage):
            cell.state = gemc_states_list[idx][lin_indx][cell_indx]

for i in range(4):
    gemc_tHMMobj_list[i].X = sort_lins(gemc_tHMMobj_list[i])


def makeFigure():
    """
    Makes figure 150.
    """

    titles = ["Control", "Gemcitabine 5 nM", "Gemcitabine 10 nM", "Gemcitabine 30 nM"]
    ax, f = getSetup((15, 35), (150, 4))

    for i in range(150):
        ax[4 * i].axis('off')
        ax[4 * i + 1].axis('off')
        ax[4 * i + 2].axis('off')
        ax[4 * i + 3].axis('off')
        plotLineage(gemc_tHMMobj_list[0].X[i], ax[4 * i])
        plotLineage(gemc_tHMMobj_list[1].X[i], ax[4 * i + 1])
        plotLineage(gemc_tHMMobj_list[2].X[i], ax[4 * i + 2])
        plotLineage(gemc_tHMMobj_list[3].X[i], ax[4 * i + 3])

    for i in range(4):
        ax[i].axis('off')
        ax[i].text(-0.2, 1.55, ascii_lowercase[i], transform=ax[i].transAxes, fontsize=20, fontweight="bold", va="top")
        ax[i].text(0.0, 1.55, titles[i], transform=ax[i].transAxes, fontsize=20, va="top")

    return f
