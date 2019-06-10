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
import matplotlib
import pdb


def makeFigure():
    # Get list of axis objects
    ax, f = getSetup((10, 10), (1, 1))
    # x1val, x2val, yval
    x1val, x2val, AIC_mean, AIC_std = Lineage_Length(T_MAS=500, T_2=100, reps=10, MASinitCells=[1], MASlocBern=[0.8], MASbeta=[80], initCells2=[1], locBern2=[0.99], beta2=[20], numStates=2, max_lin_length=200, min_lin_length=80, FOM='E', verbose=False, AIC=True, numState_start=1, numState_end=5)
    ax[0].errorbar(x1val, AIC_mean, yerr=AIC_std, marker='*', c='b', fmt= 'o')
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].grid(True, linestyle='--')
    ax[0].set_xlabel('Number of States')
    ax[0].set_ylabel('AIC Cost')
    ax[0].set_title('Akaike Information Criterion')
    #ax[0].set_ylim(1.1)
    #ax[0].subplots_adjust(top=1.3)
    ax1 = ax[0].twiny()
    ax1.set_xticks(x1val)
    ax1.set_xbound(ax1.get_xbound())
    ax1.set_xticklabels(x2val)
    ax1.set_xlabel('Number of parameters')
    matplotlib.rcParams.update({'font.size': 28})
    
    
    f.tight_layout()

    return f
