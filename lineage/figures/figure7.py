"""
This creates Figure 2.
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
import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.ticker as ticker
from .Matplot_gen import moving_average

def makeFigure():
    # Get list of axis objects
    ax, f = getSetup((12, 3), (1, 1))

    #Call function for AIC 
    
    number_of_cells_h1, accuracy_h1, bern_MAS_h1, bern_2_h1, MASlocBern, locBern2, MASbeta, beta2, betaExp_MAS_h1, betaExp_2_h1 = Lineage_Length()
    Matplot_gen(ax[0:3], number_of_cells_h1, accuracy_h1, bern_MAS_h1, bern_2_h1, MASlocBern, locBern2, MASbeta, beta2,
                betaExp_MAS_h1, betaExp_2_h1, xlabel='Cells per Lineage', FOM='E')  # Figure plots scale vs lineage length

    numb_of_lineage_h1, accuracy_h1, bern_MAS_h1, bern_2_h1, MASlocBern, locBern2, betaExp_MAS_h1, betaExp_2_h1, MASbetaExp, betaExp2 = Lineages_per_Population_Figure()
    
    
    x_vs_acc = np.column_stack((KL_h1, acc_h1))
    sorted_x_vs_acc = x_vs_acc[np.argsort(x_vs_acc[:, 0])]
    ax.set_xlabel(xlabel)
    ax.set_xscale('log')
    ax[0].set_ylim(0, 110)
    ax[0].errorbar(KL_h1, acc_h1, fmt='o', c='b', marker="*", fillstyle='none', label='Accuracy')
    ax[0].plot(sorted_x_vs_acc[:, 0][9:], moving_average(sorted_x_vs_acc[:, 1]), c='k', label='Moving Average')
    ax[0].axhline(y=100, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='b')  # linestyle is dashdotdotted
    ax[0].set_ylabel('Accuracy (%)', rotation=90)
    ax[0].get_yticks()
    ax[0].set_title('Accuracy')
    ax[0].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.0e'))
    
    Matplot_gen(ax[3:6], numb_of_lineage_h1, accuracy_h1, bern_MAS_h1, bern_2_h1, MASlocBern, locBern2, MASbeta, beta2,
                betaExp_MAS_h1, betaExp_2_h1, xlabel='Lineages per Population')  # Figure plots scale vs number of lineages

    f.tight_layout()

    return f
