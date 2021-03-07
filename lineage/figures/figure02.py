"""
Handful of lineages in figure 12. 
"""
from string import ascii_lowercase
import pickle

from .figureCommon import getSetup
from ..plotTree import plotLineage

# open lapatinib
pik1 = open("gemcitabines.pkl", "rb")
gemc_tHMMobj_list = []
for _ in range(4):
    gemc_tHMMobj_list.append(pickle.load(pik1))

for i, thmmobj in enumerate(gemc_tHMMobj_list):
    st1 = []
    st2 = []
    st3 = []
    for lins in thmmobj.X:
        if lins.output_lineage[0].state == 0:
            st1.append(lins)
        elif lins.output_lineage[0].state == 1:
            st2.append(lins)
        else:
            st3.append(lins)
    thmmobj.X = st1[11:13] + st2[1:3] + st3[15:17]

def makeFigure():
    """
    Makes figure 12 lineage.
    """
    ax, f = getSetup((12, 2), (len(gemc_tHMMobj_list[1].X), 4))
    k = 0
    for i in range(6):
        for objs in gemc_tHMMobj_list:
            ax[k].axis('off')
            plotLineage(objs.X[i], ax[k])
            k += 1

    return f
