"""
This creates Figure 2.
"""
from .figureCommon import subplotLabel, getSetup


def makeFigure():
    # Get list of axis objects
    ax, f = getSetup((9, 3), (1, 3))

    '''number_of_cells_h1, accuracy_h1, bern_MAS_h1, bern_2_h1, MASlocBern, locBern2, MASbeta, beta2, betaExp_MAS_h1, betaExp_2_h1 = Lineage_Length()
    Matplot_gen(ax[0:3], number_of_cells_h1, accuracy_h1, bern_MAS_h1, bern_2_h1, MASlocBern, locBern2, MASbeta, beta2,
                betaExp_MAS_h1, betaExp_2_h1, xlabel='Cells per Lineage', FOM='E')  # Figure plots scale vs lineage length'''

    numb_of_lineage_h1, accuracy_h1, bern_MAS_h1, bern_2_h1, MASlocBern, locBern2, betaExp_MAS_h1, betaExp_2_h1, MASbeta, beta2 = Lineages_per_Population_Figure()
    Matplot_gen(ax[0:3], numb_of_lineage_h1, accuracy_h1, bern_MAS_h1, bern_2_h1, MASlocBern, locBern2, MASbeta, beta2,
                betaExp_MAS_h1, betaExp_2_h1, xlabel='Lineages per Population')  # Figure plots scale vs number of lineages
    f.tight_layout()

    return f

