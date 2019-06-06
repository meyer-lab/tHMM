"""
This creates Figure 7. AIC Figure.
"""
from .figureCommon import subplotLabel, getSetup
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker as ticker
from .Depth_Two_State_Lineage import Depth_Two_State_Lineage
from ..Analyze import Analyze
from .Matplot_gen import Matplot_gen
from .Fig_Gen import Lineage_Length, Lineages_per_Population_Figure
from ..tHMM_utils import getAccuracy, getAIC
from ..Lineage_utils import remove_singleton_lineages
from matplotlib.ticker import MaxNLocator
import pdb


def makeFigure():
    # Get list of axis objects
    ax, f = getSetup((10, 10), (1, 2))
    # x1val, x2val, yval
    x1val, x2val, yval = Lineage_Length(T_MAS=500, T_2=100, reps=1, MASinitCells=[1], MASlocBern=[0.999], MASbeta=[80], initCells2=[1],
                   locBern2=[0.8], beta2=[20], numStates=2, max_lin_length=300, min_lin_length=5, FOM='E', verbose=False, AIC=True)
    pdb.set_trace()
    ax[0].scatter(x1val, yval, marker='*', c='b', s=500, label='One state data/model')
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].grid(True, linestyle='--')
    ax[0].set_xlabel('Number of States')
    ax[0].set_ylabel('AIC Cost')
    ax[0].set_title('Akaike Information Criterion')
    #ax[0].set_ylim(1.1)
    #ax[0].subplots_adjust(top=1.3)

    ax[1] = ax[0].twiny()
    ax[1].set_xticks([1]+ax[1].get_xticks())
    ax[1].set_xbound(ax[1].get_xbound())
    ax[1].set_xticklabels(x2val)
    ax[1].set_xlabel('Number of parameters')

    ax[0].legend()
    #rcParams.update({'font.size': 28})
    
    
    f.tight_layout()

    return f
