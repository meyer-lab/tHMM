""" In this file we plot! """

import numpy as np
from Bio.Phylo.BaseTree import Clade
from Bio import Phylo
from matplotlib import pylab


def CladeRecursive(cell, a, censore):
    """ To plot the lineage while censored (from G1 or G2).
    If cell died in G1, the lifetime of the cell until dies is shown in red.
    If cell died in G2, the lifetime of the cell until dies is shown in blue.
    If none of the above, the cell continues to divide and is shown in black.
    a should be: a = [Clade(lineage1.full_lineage[0].obs[2]+lineage1.full_lineage[0].obs[3])]
    If you are interested, you can take a look at the source code for creating Clades manually:
    https://github.com/biopython/biopython/blob/fce4b11b4b8e414f1bf093a76e04a3260d782905/Bio/Phylo/BaseTree.py#L801
    """
    if cell.state == 0:
        colorr = "olive"
    elif cell.state == 1:
        colorr = "salmon"
    elif cell.state == 2:
        colorr = "red"
    else:
        colorr = "black"
    # pink lines mean the cell was in G1 when died or experiment ended
    # gold lines mean the cell was in G2 when died or experiment ended

    if cell.isLeaf() and censore:
        if cell.obs[0] == 0:  # the cell died in G1
            assert np.isfinite(cell.obs[2])
            return Clade(branch_length=cell.obs[2], width=1, color="pink")
        elif cell.obs[0] == 1 and cell.obs[1] == 0: # if cell dies in G2
            assert np.isfinite(cell.obs[2]+cell.obs[3])
            return Clade(branch_length=cell.obs[2]+cell.obs[3], width=1, color="gold")
        elif cell.obs[0] == 1 and np.isnan(cell.obs[1]): # cell stays in G1 until the end
            assert np.isfinite(cell.obs[2])
            return Clade(branch_length=cell.obs[2], width=1, color="pink")

    else:
        clades = []
        if cell.left is not None and cell.left.observed:
            clades.append(CladeRecursive(cell.left, a, censore))
        if cell.right is not None and cell.right.observed:
            clades.append(CladeRecursive(cell.right, a, censore))
        return Clade(branch_length=cell.obs[2]+cell.obs[3], width=1, clades=clades, color=colorr)


def plotLineage(lineage, axes, censore=True):
    """
    Makes lineage tree.
    """

    a = [Clade(lineage.output_lineage[0].time)]

    # input the root cells in the lineage
    c = CladeRecursive(lineage.output_lineage[0], a, censore)

    return Phylo.draw(c, axes=axes)
