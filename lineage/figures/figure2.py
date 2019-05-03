"""
This creates Figure 2.
"""
from .figureCommon import subplotLabel, getSetup
import numpy as np
from matplotlib import pyplot as plt
from .Depth_Two_State_Lineage import Depth_Two_State_Lineage
from ..Analyze import Analyze
from .Matplot_gen import Matplot_gen
from .Fig_Gen import Lineage_Length, Lineages_per_Population_Figure
from ..tHMM_utils import getAccuracy, getAIC
from ..Lineage_utils import remove_singleton_lineages


def makeFigure():
    # Get list of axis objects
    x, y = 3, 1  # rows and columns
    ax, f = getSetup((3, 6), (x, y))

    x, acc_h1, bern_MAS_h1, bern_2_h1, MASlocBern, locBern2, MASbeta, beta2, betaExp_MAS_h1, betaExp_2_h1 = Lineage_Length()
    Matplot_gen(ax[0:3], x, acc_h1, bern_MAS_h1, bern_2_h1, MASlocBern, locBern2, MASbeta, beta2, xlabel='Cells per Lineage', FOM='E')  # Figure plots scale vs lineage length

    f.tight_layout()

    return f
