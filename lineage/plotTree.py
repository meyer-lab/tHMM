""" In this file we plot! """

from Bio.Phylo.BaseTree import Clade
from Bio import Phylo

def fullRecursive(cell, a):
    """ To plot the <full> binary tree based on inter-mitotic time (lifetime) of cells.
    a should be: a = [Clade(lineage1.full_lineage[0].obs[2]+lineage1.full_lineage[0].obs[3])]
    """
    if cell.isLeaf():
        return Clade(branch_length=(cell.time))
    else:
        return Clade(branch_length=(cell.time), clades=[fullRecursive(cell.left, a), fullRecursive(cell.right, a)])

def CensoredRecursive(cell, a):
    """ To plot the lineage while censored (from G1 or G2).
    If cell died in G1, the lifetime of the cell until dies is shown in red.
    If cell died in G2, the lifetime of the cell until dies is shown in blue.
    If none of the above, the cell continues to divide and is shown in black.
    a should be: a = [Clade(lineage1.full_lineage[0].obs[2]+lineage1.full_lineage[0].obs[3])]
    """
    if cell.isLeaf():
        return Clade(branch_length=(cell.time))
    else:
        if cell.obs[0] == 0:
            return Clade(branch_length=(cell.time), color="red") # dead in G1
        elif cell.obs[1] == 0:
            return Clade(branch_length=(cell.time), color="blue") # dead in G2
        else:
            return Clade(branch_length=(cell.time),
                         clades=[CensoredRecursive(cell.left, a), CensoredRecursive(cell.right, a)])

def plotLineage(lineage, path):
    """
    Makes lineage tree.
    """

    a = [Clade(lineage.full_lineage[0].time)]

    # input the root cells in the lineage
    c = CensoredRecursive(lineage.full_lineage[0], a)

    Phylo.draw(c)
    pylab.axis('off')
    pylab.savefig(path, format='svg', bbox_inches='tight', dpi=300)
