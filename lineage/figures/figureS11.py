""" This is a file to put together 4 conditions of lapatinib together. """

from string import ascii_lowercase
import pickle
from .common import getSetup, sort_lins
from ..plotTree import plotLineage


pik1 = open("lapatinibs.pkl", "rb")
alls = []
for i in range(7):
    lapt_tHMMobj_list = []
    for i in range(4):
        lapt_tHMMobj_list.append(pickle.load(pik1))
    alls.append(lapt_tHMMobj_list)

lapt_tHMMobj_list = alls[3]
assert len(lapt_tHMMobj_list) == 4

lapt_states_list = [tHMMobj.predict() for tHMMobj in lapt_tHMMobj_list]

# assign the predicted states to each cell
for idx, lapt_tHMMobj in enumerate(lapt_tHMMobj_list):
    for lin_indx, lin in enumerate(lapt_tHMMobj.X):
        for cell_indx, cell in enumerate(lin.output_lineage):
            cell.state = lapt_states_list[idx][lin_indx][cell_indx]

for i in range(4):
    lapt_tHMMobj_list[i].X = sort_lins(lapt_tHMMobj_list[i])
    print(len(lapt_tHMMobj_list[i].X))


def makeFigure():
    """
    Makes figure 100.
    """
    titles = ["Control", "Lapatinib 25 nM", "Lapatinib 50 nM", "Lapatinib 250 nM"]
    ax, f = getSetup((15, 65), (160, 4))

    for i in range(160):
        ax[4 * i].axis('off')
        ax[4 * i + 1].axis('off')
        ax[4 * i + 2].axis('off')
        ax[4 * i + 3].axis('off')
    for i in range(len(lapt_tHMMobj_list[0].X)):
        plotLineage(lapt_tHMMobj_list[0].X[i], ax[4 * i])
    for i in range(len(lapt_tHMMobj_list[1].X)):
        plotLineage(lapt_tHMMobj_list[1].X[i], ax[4 * i + 1])
    for i in range(len(lapt_tHMMobj_list[2].X)):
        plotLineage(lapt_tHMMobj_list[2].X[i], ax[4 * i + 2])
    for i in range(len(lapt_tHMMobj_list[3].X)):
        plotLineage(lapt_tHMMobj_list[3].X[i], ax[4 * i + 3])

    for i in range(4):
        ax[i].axis('off')
        ax[i].text(-0.2, 1.55, ascii_lowercase[i], transform=ax[i].transAxes, fontsize=20, fontweight="bold", va="top")
        ax[i].text(0.0, 1.55, titles[i], transform=ax[i].transAxes, fontsize=20, va="top")

    return f
