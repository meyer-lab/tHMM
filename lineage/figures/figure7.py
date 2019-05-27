"""
This creates Figure 7. AIC Figure.
"""
from .figureCommon import subplotLabel, getSetup
import numpy as np
from matplotlib import pyplot as plt
from .Depth_Two_State_Lineage import Depth_Two_State_Lineage
from ..Analyze import Analyze
from .Matplot_gen import Matplot_gen
from .Fig_Gen import Lineage_Length, Lineages_per_Population_Figure, AIC
from ..tHMM_utils import getAccuracy, getAIC
from ..Lineage_utils import remove_singleton_lineages
from matplotlib.ticker import MaxNLocator


def makeFigure():
    # Get list of axis objects
    ax, f = getSetup((10, 10), (1, 2))

    #Call function for AIC 
    
    x1val = []
    x2val = []
    yval = []
    
    #need to generate X, and do this for 
    for numState in range(3):
        tHMMobj = tHMM(X, numStates=numState, FOM='G') # build the tHMM class with X
        tHMMobj, NF, betas, gammas, LL = fit(tHMMobj, max_iter=100, verbose=False)
        AIC_value, numStates, deg = getAIC(tHMMobj, LL)
        x1val.append(numStates)
        x2val.append(deg)
        yval.append(AIC_value)

    ax[0].scatter(xval, yval, marker='*', c='b', s=500, label='One state data/model')
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].grid(True, linestyle='--')
    ax[0].set_xlabel('Number of States')
    ax[0].set_ylabel('AIC Cost')
    ax[0].set_title('Akaike Information Criterion')
    ax[0].set_y(1.1)
    ax[0].subplots_adjust(top=1.3)

    ax[1] = ax[0].twiny()
    ax[1].set_xticks([1]+ax1.get_xticks())
    ax[1].set_xbound(ax1.get_xbound())
    ax[1].set_xticklabels(x2val)
    ax[1].set_xlabel('Number of parameters')

    ax[0].legend()
    rcParams.update({'font.size': 28})
    
    
    f.tight_layout()

    return f
