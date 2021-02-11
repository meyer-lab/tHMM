""" This file plots the trees with their predicted states for lapatinib. """

import pickle

from .figureCommon import getSetup
from ..plotTree import plotLineage

# open gemcitabine
pik1 = open("gemcitabines.pkl", "rb")
gemc_tHMMobj_list = []
for _ in range(4):
    gemc_tHMMobj_list.append(pickle.load(pik1))


st1 = []
st2 = []
st3 = []
for lins in gemc_tHMMobj_list[2].X:
    if lins.output_lineage[0].state == 0:
        st1.append(lins)
    elif lins.output_lineage[0].state == 1:
        st2.append(lins)
    else:
        st3.append(lins)
gemc_tHMMobj_list[2].X = st1[0:min(50, len(st1))] + st2[0:min(50, len(st2))] + st3[0:min(50, len(st3))]


def makeFigure():
    """
    Makes figure 152.
    """
    ax, f = getSetup((7, 40), (len(gemc_tHMMobj_list[2].X), 1))

    for i, X in enumerate(gemc_tHMMobj_list[2].X):
        ax[i].axis('off')
        plotLineage(X, ax[i])
    return f
