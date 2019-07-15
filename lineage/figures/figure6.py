"""
This creates Figure 6.
"""
from .figureCommon import getSetup
import numpy as np
from matplotlib import ticker as ticker
from .Fig_Gen import KL_per_lineage
from .Matplot_gen import moving_average


def makeFigure():
    """ Main figure generating function for Fig. 6 """
    ax, f = getSetup((7, 7), (1, 1))

    KL_h1, acc_h1, _, _, _, _, _, _, _, _ = KL_per_lineage()

    x_vs_acc = np.column_stack((KL_h1, acc_h1))
    sorted_x_vs_acc = x_vs_acc[np.argsort(x_vs_acc[:, 0])]
    ax[0].set_xlabel('KL Divergence')
    ax[0].set_xscale('log')
    ax[0].set_ylim(0, 110)
    ax[0].errorbar(KL_h1, acc_h1, fmt='o', c='b', marker="*", fillstyle='none', label='Accuracy', alpha=0.5)
    ax[0].plot(sorted_x_vs_acc[:, 0][9:], moving_average(sorted_x_vs_acc[:, 1]), c='k', label='Moving Average')
    ax[0].axhline(y=100, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='b')  # linestyle is dashdotdotted
    ax[0].set_ylabel('Accuracy (%)', rotation=90)
    ax[0].get_yticks()
    ax[0].set_title('Effect of Subpopulation Similarity on Accuracy')
    ax[0].xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0e'))

    f.tight_layout()

    return f
