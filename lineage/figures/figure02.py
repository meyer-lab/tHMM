"""
Handful of lineages in figure 11.
"""
import pickle

from .common import getSetup
from ..plotTree import plotLineage

# open lapatinib
pik1 = open("gemcitabines.pkl", "rb")
gemc_tHMMobj_list = []
for _ in range(4):
    gemc_tHMMobj_list.append(pickle.load(pik1))


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
