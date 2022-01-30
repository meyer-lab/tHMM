"""
Handful of lineages in figure 91.
"""
import pickle
from .common import getSetup
from ..plotTree import plotLineage_MCF10A

pik1 = open("gf.pkl", "rb")
gf_tHMMobj_list = []
for i in range(4):
    gf_tHMMobj_list.append(pickle.load(pik1))


def makeFigure():
    """
    Makes figure 91 lineage.
    """
    ax, f = getSetup((12, 2), (6, 4))
    k = 0
    for i in range(6):
        for objs in gf_tHMMobj_list:
            ax[k].axis('off')
            plotLineage_MCF10A(objs.X[i], ax[k])
            k += 1
    return f
