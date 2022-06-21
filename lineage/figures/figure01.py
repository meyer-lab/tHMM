"""
Handful of lineages in figure 11.
"""
import pickle

from .common import getSetup
from ..plotTree import plotLineage

# open lapatinib
pik1 = open("lapatinibs.pkl", "rb")
alls = []
for i in range(7):
    lapt_tHMMobj_list = []
    for i in range(4):
        lapt_tHMMobj_list.append(pickle.load(pik1))
    alls.append(lapt_tHMMobj_list)

# selected for lapatinib is 4 states which is index 3.
lapt_tHMMobj_list = alls[3]

lapt_states_list = [tHMMobj.predict() for tHMMobj in lapt_tHMMobj_list]

# assign the predicted states to each cell
for idx, lapt_tHMMobj in enumerate(lapt_tHMMobj_list):
    for lin_indx, lin in enumerate(lapt_tHMMobj.X):
        for cell_indx, cell in enumerate(lin.output_lineage):
            cell.state = lapt_states_list[idx][lin_indx][cell_indx]

def makeFigure():
    """
    Makes figure 11 lineage.
    """
    ax, f = getSetup((12, 2), (6, 4))
    k = 0
    for i in range(6):
        for objs in lapt_tHMMobj_list:
            ax[k].axis('off')
            plotLineage(objs.X[i], ax[k])
            k += 1

    return f
