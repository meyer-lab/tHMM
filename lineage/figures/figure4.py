"""
This creates Figure 4.
"""
from .figureCommon import getSetup
from .Matplot_gen import Matplot_gen
from .Fig_Gen import Lineage_Length


def makeFigure():
    # Get list of axis objects
    ax, f = getSetup((12, 3), (1, 3))

    number_of_cells_h1, accuracy_h1, bern_MAS_h1, bern_2_h1, MASlocBern, locBern2, MASbeta, beta2, betaExp_MAS_h1, betaExp_2_h1 = Lineage_Length()
    Matplot_gen(ax[0:3], number_of_cells_h1, accuracy_h1, bern_MAS_h1, bern_2_h1, MASlocBern, locBern2, MASbeta, beta2,
                betaExp_MAS_h1, betaExp_2_h1, xlabel='Cells per Lineage', FOM='E')  # Figure plots scale vs lineage length'''

    f.tight_layout()

    return f