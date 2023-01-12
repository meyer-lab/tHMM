""" In this file we plot the raw data before fitting from paclitaxel treated HCC1143 cells """
from ..Lineage_collections import taxols as Taxol_lin_list
import numpy as np
from ..import_lineage import import_taxol
from ..plotTree import plotLineage
from .common import getSetup
from string import ascii_lowercase
from ..states.StateDistributionGaPhs import StateDistribution
from ..LineageTree import LineageTree

desired_num_states = 2
E = [StateDistribution() for _ in range(desired_num_states)]

untreated_t, taxol_05, taxol_1, taxol_15, taxol_2, taxol_25, taxol_3, taxol_4 = import_taxol()

counts = [len(i) for i in Taxol_lin_list]
print(counts)

def makeFigure():
    """
    Makes figure S18.
    """
    titles = ["Untreated", "Taxol 1 nM", "Taxol 2 nM", "Taxol 3 nM", "Taxol 4 nM"]
    ax, f = getSetup((20, 35), (np.min(counts), 5))

    for i in range(np.min(counts)):
        ax[5 * i].axis('off')
        ax[5 * i + 1].axis('off')
        ax[5 * i + 2].axis('off')
        ax[5 * i + 3].axis('off')
        ax[5 * i + 4].axis('off')
        plotLineage(Taxol_lin_list[0][i], ax[5 * i], color=False)
        plotLineage(Taxol_lin_list[1][i], ax[5 * i + 1], color=False)
        plotLineage(Taxol_lin_list[2][i], ax[5 * i + 2], color=False)
        plotLineage(Taxol_lin_list[3][i], ax[5 * i + 3], color=False)
        plotLineage(Taxol_lin_list[4][i], ax[5 * i + 4], color=False)

    for i in range(5):
        ax[i].axis('off')
        ax[i].text(-0.2, 1.55, ascii_lowercase[i], transform=ax[i].transAxes, fontsize=20, fontweight="bold", va="top")
        ax[i].text(0.0, 1.55, titles[i], transform=ax[i].transAxes, fontsize=20, va="top")

    return f