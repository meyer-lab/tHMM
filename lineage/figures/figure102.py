""" This file plots the trees with their predicted states for lapatinib. """

import pickle

from .figureCommon import getSetup
from ..plotTree import plotLineage

# open lapatinib
pik1 = open("lapatinibs.pkl", "rb")
lapt_tHMMobj_list = []
for _ in range(4):
    lapt_tHMMobj_list.append(pickle.load(pik1))

st1 = []
st2 = []
st3 = []
for lins in lapt_tHMMobj_list[2].X:
    if lins.output_lineage[0].state == 0:
        st1.append(lins)
    elif lins.output_lineage[0].state == 1:
        st2.append(lins)
    else:
        st3.append(lins)

lapt_tHMMobj_list[2].X = st1[0:min(25, len(st1))] + st2[0:min(25, len(st2))] + st3[0:min(25, len(st3))]


def makeFigure():
    """
    Makes figure 102.
    """
    ax, f = getSetup((7, 40), (len(lapt_tHMMobj_list[2].X), 1))

    for i, X in enumerate(lapt_tHMMobj_list[2].X):
        ax[i].axis('off')
        plotLineage(X, ax[i])
    return f
