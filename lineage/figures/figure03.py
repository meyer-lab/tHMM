"""
Handful of lineages in figure 91.
"""
import pickle
from .common import getSetup
from ..plotTree import plotLineage_MCF10A

pik1 = open("gf.pkl", "rb")
alls = []
for i in range(7):
    hgf_tHMMobj_list = []
    for i in range(4):
        hgf_tHMMobj_list.append(pickle.load(pik1))
    alls.append(hgf_tHMMobj_list)

# selected for gf treatments is 3 states which is index 2.
hgf_tHMMobj_list = alls[2]

hgf_states_list = [tHMMobj.predict() for tHMMobj in hgf_tHMMobj_list]

# assign the predicted states to each cell
for idx, hgf_tHMMobj in enumerate(hgf_tHMMobj_list):
    for lin_indx, lin in enumerate(hgf_tHMMobj.X):
        for cell_indx, cell in enumerate(lin.output_lineage):
            cell.state = hgf_states_list[idx][lin_indx][cell_indx]


def makeFigure():
    """
    Makes figure 91 lineage.
    """
    ax, f = getSetup((12, 2), (8, 4))
    k = 0
    for i in range(8):
        for objs in hgf_tHMMobj_list:
            ax[k].axis('off')
            plotLineage_MCF10A(objs.X[i], ax[k])
            k += 1
    return f
