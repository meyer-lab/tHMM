"""
This creates Figure 2.
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
    x, y = 1, 2  # rows and columns
    ax, f = getSetup((12, 9), (x, y))

    '''x, acc_h1, bern_MAS_h1, bern_2_h1, MASlocBern, locBern2, cGom_MAS_h1, cGom_2_h1, MAScGom, cGom2, scaleGom_MAS_h1, scaleGom_2_h1, MASscaleGom, scaleGom2 = Lineage_Length()
    Matplot_gen(ax[0:4], x, acc_h1, bern_MAS_h1, bern_2_h1, MASlocBern, locBern2, cGom_MAS_h1, cGom_2_h1, MAScGom, cGom2, scaleGom_MAS_h1, scaleGom_2_h1, MASscaleGom, scaleGom2, xlabel='Cells per Lineage')  # Figure plots scale vs lineage length

    x, acc_h1, bern_MAS_h1, bern_2_h1, MASlocBern, locBern2, cGom_MAS_h1, cGom_2_h1, MAScGom, cGom2, scaleGom_MAS_h1, scaleGom_2_h1, MASscaleGom, scaleGom2 = Lineages_per_Population_Figure()
    Matplot_gen(ax[4:8], x, acc_h1, bern_MAS_h1, bern_2_h1, MASlocBern, locBern2, cGom_MAS_h1, cGom_2_h1, MAScGom, cGom2, scaleGom_MAS_h1, scaleGom_2_h1, MASscaleGom, scaleGom2, xlabel='Lineages per Population')  # Figure plots scale vs number of lineages'''
    
    x, acc_h1, bern_MAS_h1, bern_2_h1, MASlocBern, locBern2, cGom_MAS_h1, cGom_2_h1, MAScGom, cGom2, scaleGom_MAS_h1, scaleGom_2_h1, MASscaleGom, scaleGom2 = KL_per_lineage()
    Matplot_gen_KL(ax[0:2], x, acc_h1, bern_MAS_h1, bern_2_h1, MASlocBern, locBern2, cGom_MAS_h1, cGom_2_h1, MAScGom, cGom2, scaleGom_MAS_h1, scaleGom_2_h1, MASscaleGom, scaleGom2, xlabel='KL Divergence')  # Figure plots scale vs KL

    f.tight_layout()

    return f
