"""
Handful of lineages in figure 11.
"""
import pickle

from .common import getSetup
from ..plotTree import plotLineage

# open lapatinib
pik1 = open("gemcitabines.pkl", "rb")
alls = []
for i in range(7):
    gemc_tHMMobj_list = []
    for i in range(4):
        gemc_tHMMobj_list.append(pickle.load(pik1))
    alls.append(gemc_tHMMobj_list)

# selected for lapatinib is 5 states which is index 4.
gemc_tHMMobj_list = alls[4]

gemc_states_list = [tHMMobj.predict() for tHMMobj in gemc_tHMMobj_list]

for idx, gemc_tHMMobj in enumerate(gemc_tHMMobj_list):
    for lin_indx, lin in enumerate(gemc_tHMMobj.X):
        for cell_indx, cell in enumerate(lin.output_lineage):
            cell.state = gemc_states_list[idx][lin_indx][cell_indx]

def makeFigure():
    """
    Makes figure 12 lineage.
    """
    ax, f = getSetup((12, 2), (6, 4))
    k = 0
    for i in range(6):
        for objs in gemc_tHMMobj_list:
            ax[k].axis('off')
            plotLineage(objs.X[i], ax[k])
            k += 1

    return f
