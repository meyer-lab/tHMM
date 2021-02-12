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
for thmmObj in gemc_tHMMobj_list:
    for lins in thmmObj.X:
        if lins.output_lineage[0].state == 0:
            st1.append(lins)
        elif lins.output_lineage[0].state == 1:
            st2.append(lins)
        else:
            st3.append(lins)
    thmmObj.X = st1[-51:-1] + st2[-51:-1] + st3[-51:-1]


def makeFigure():
    """
    Makes figure 15.
    """
    ax, f = getSetup((7, 40), (150, 1))

    for i in range(150):
        ax[i].axis('off')
        plotLineage(gemc_tHMMobj_list[0].X[i], ax[i])

    return f
