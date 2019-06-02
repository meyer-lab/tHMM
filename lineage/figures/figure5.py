"""
This creates Figure 5.
"""
from .figureCommon import subplotLabel, getSetup
from .Matplot_gen import Matplot_gen
from .Fig_Gen import Lineages_per_Population_Figure


def makeFigure():
    # Get list of axis objects
    ax, f = getSetup((12, 3), (1, 3))

    numb_of_lineage_h1, accuracy_h1, bern_MAS_h1, bern_2_h1, MASlocBern, locBern2, MASbeta, beta2, betaExp_MAS_h1, betaExp_2_h1 = Lineages_per_Population_Figure()
    Matplot_gen(ax[0:3], numb_of_lineage_h1, accuracy_h1, bern_MAS_h1, bern_2_h1, MASlocBern, locBern2, MASbeta, beta2,
                betaExp_MAS_h1, betaExp_2_h1, xlabel='Lineages per Population', FOM='E')  # Figure plots scale vs number of lineages

    f.tight_layout()

    return f