""" This file plots the trees with their predicted states for growth factors. """

import pickle
import numpy as np
from .common import getSetup, sort_lins
from ..plotTree import plotLineage_MCF10A

# open lapatinib
pik1 = open("gf.pkl", "rb")
gf_tHMMobj_list = []
for _ in range(4):
    gf_tHMMobj_list.append(pickle.load(pik1))


for thmm_obj in gf_tHMMobj_list:
    thmm_obj.X = sort_lins(thmm_obj)

conditions = ["PBS", "EGF", "HGF", "OSM"]

def makeFigure():
    """
    Makes figure 101.
    """
    num_lins = [len(gf_tHMMobj_list[i].X) for i in range(4)]
    ax, f = getSetup((10, 20), (np.max(num_lins), 4))

    for i in range(4*np.max(num_lins)):
        ax[i].axis('off')

    for j, thmmobj in enumerate(gf_tHMMobj_list):
        for i, X in enumerate(thmmobj.X):
            plotLineage_MCF10A(X, ax[4*i+j])

    for i in range(4):
        ax[i].set_title(conditions[i])

    return f
