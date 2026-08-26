"""In this file we plot the raw data before fitting from paclitaxel treated HCC1143 cells"""

# from ..Lineage_collections import taxols as Taxol_lin_list
from string import ascii_lowercase

from ..import_lineage import import_taxol_file
from ..LineageTree import LineageTree
from ..plotTree import plotLineage
from ..states.StateDistributionGaPhs import StateDistribution
from .common import getSetup

desired_num_states = 2
E = [StateDistribution() for _ in range(desired_num_states)]
untreated_t = import_taxol_file()
untreated_taxol = [LineageTree(list_cells, E) for list_cells in untreated_t]
c = 0
for lins in untreated_t:
    for _cell in lins:
        c += 1
# counts = len(untreated_taxol)
counts = 100
print("number of lineages", len(untreated_taxol), " and the number of total cells: ", c)


def makeFigure():
    """
    Makes figure S18.
    """
    titles = ["Untreated", "Taxol 1 nM", "Taxol 2 nM", "Taxol 3 nM", "Taxol 4 nM"]
    ax, f = getSetup((17, 35), (counts, 4))

    for i in range(counts):
        ax[4 * i].axis("off")
        # ax[4 * i + 1].axis('off')
        # ax[4 * i + 2].axis('off')
        # ax[4 * i + 3].axis('off')
        plotLineage(untreated_taxol[i], ax[4 * i], color=False)
        # plotLineage(Taxol_lin_list[1][i], ax[4 * i + 1], color=False)
        # plotLineage(Taxol_lin_list[2][i], ax[4 * i + 2], color=False)
        # plotLineage(Taxol_lin_list[3][i], ax[4 * i + 3], color=False)

    for i in range(4):
        ax[i].axis("off")
        ax[i].text(-0.2, 1.55, ascii_lowercase[i], transform=ax[i].transAxes, fontsize=20, fontweight="bold", va="top")
        ax[i].text(0.0, 1.55, titles[i], transform=ax[i].transAxes, fontsize=20, va="top")

    return f
