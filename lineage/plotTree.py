""" In this file we plot! """

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
    else:
        colorr = "salmon"

    if cell.isLeaf() and censore:
        if cell.time.transition_time >= cell.time.endT:  # the cell died in G1
            return Clade(branch_length=cell.time.endT - cell.time.startT, color="pink")
        else:  # the cell spent some time in G2
            return Clade(branch_length=cell.time.endT - cell.time.startT, color="gold")  # dead in G2
    else:
        clades = []
        if cell.left is not None and cell.left.observed:
            clades.append(CladeRecursive(cell.left, a, censore))
        if cell.right is not None and cell.right.observed:
            clades.append(CladeRecursive(cell.right, a, censore))
        return Clade(branch_length=cell.time.endT - cell.time.startT, clades=clades, color=colorr)


def plotLineage(lineage, path, censore=True):
    """
    Makes lineage tree.
    """

    a = [Clade(lineage.output_lineage[0].time.endT)]

    # input the root cells in the lineage
    c = CladeRecursive(lineage.output_lineage[0], a, censore)

    Phylo.draw(c)
    pylab.axis("off")
    pylab.savefig(path, format="svg", transparent=True, bbox_inches="tight", dpi=400)
