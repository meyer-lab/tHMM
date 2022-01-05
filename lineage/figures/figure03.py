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
    Makes figure 12 lineage.
    """
    ax, f = getSetup((12, 4), (14, 3))
    k = 0
    for i in range(6):
        for objs in gf_tHMMobj_list[0:3]:
            ax[k].axis('off')
            plotLineage_MCF10A(objs.X[i], ax[k])
            k += 1

    k = 24
    for i in range(6):
        ax[k].axis('off')
        ax[k+1].axis('off')
        ax[k+2].axis('off')
        plotLineage_MCF10A(gf_tHMMobj_list[3].X[i], ax[k])
        k += 3

    for j in range(18, 24):
        ax[j].axis('off')
    return f
