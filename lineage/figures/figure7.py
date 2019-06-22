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
    ax, f = getSetup((5, 5), (1, 1))
    # x1val, x2val, yval
    numstateval, AIC_mean, AIC_std, LL_mean, LL_std = Lineage_Length(T_MAS=500, T_2=100, reps=10, MASinitCells=[1], MASlocBern=[0.8], MASbeta=[80], initCells2=[1], locBern2=[0.99], beta2=[20], numStates=4, max_lin_length=200, min_lin_length=80, FOM='E', verbose=False, AIC=True, numState_start=1, numState_end=2)
    ax[0].errorbar(numstateval, AIC_mean, yerr=AIC_std, c='b', fmt='-', label='AIC')
    ax[0].errorbar(numstateval, LL_mean, yerr=LL_std, c='tab:orange', fmt='-', label='Likelihood')
    ax[0].grid(True, linestyle='--')
    ax[0].set_xlabel('Number of States')
    ax[0].set_ylabel('AIC Cost')
    ax[0].set_title('Akaike Information Criterion')
    ax[0].legend(loc='best')
    f.tight_layout()

    return f
