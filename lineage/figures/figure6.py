"""
This creates Figure 6.
"""
from .figureCommon import subplotLabel, getSetup
import numpy as np
from matplotlib import pyplot as plt
from .Depth_Two_State_Lineage import Depth_Two_State_Lineage
from ..Analyze import Analyze
from .Matplot_gen import Matplot_gen
from .Matplot_gen_KL import Matplot_gen_KL
from .Fig_Gen import Lineage_Length, Lineages_per_Population_Figure, KL_per_lineage
from ..tHMM_utils import getAccuracy, getAIC
from ..Lineage_utils import remove_singleton_lineages


def makeFigure():
    # Get list of axis objects

    ax, f = getSetup((12, 3), (1, 3))

    KL_h1, acc_h1, bern_MAS_h1, bern_2_h1, MASlocBern, locBern2, MASbeta, beta2, betaExp_MAS_h1, betaExp_2_h1 = KL_per_lineage()
    Matplot_gen_KL(ax[0:3], KL_h1, acc_h1, bern_MAS_h1, bern_2_h1, MASlocBern, locBern2, MASbeta, beta2, betaExp_MAS_h1, betaExp_2_h1, xlabel='KL Divergence')

    f.tight_layout()

    return f
