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
st4 = []
st5 = []
st6 = []

for lins in gemc_tHMMobj_list[0].X:
    if lins.output_lineage[0].state == 0:
        st1.append(lins)
    elif lins.output_lineage[0].state == 1:
        st2.append(lins)
    elif lins.output_lineage[0].state == 2:
        st3.append(lins)
    elif lins.output_lineage[0].state == 3:
        st4.append(lins)
    elif lins.output_lineage[0].state == 4:
        st5.append(lins)
    elif lins.output_lineage[0].state == 5:
        st6.append(lins)
    else:
        st3.append(lins)
    gemc_tHMMobj_list[0].X = st1[-11:-1] + st2[-11:-1] + st3[-11:-1] + st4[-11:-1] + st5[-11:-1] + st6[-11:-1]


def makeFigure():
    """
    Makes figure 15.
    """
    ax, f = getSetup((7, 40), (len(gemc_tHMMobj_list[0].X), 1))

    for i in range(len(gemc_tHMMobj_list[0].X)):
        ax[i].axis('off')
        plotLineage(gemc_tHMMobj_list[0].X[i], ax[i])

    return f
