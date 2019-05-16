'''Generates a plot with 4 subplots for accuracy estimation and estimation of a bernoulli and 2 gompertz parameters. Note: this is only used for the Lineage Length and Lineage Number figures, the AIC only requires one figure of accuracy so it has its own maplotlib code  in the Fig_Gen function.'''

from matplotlib import pyplot as plt
import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.ticker as ticker


def Matplot_gen_KL(ax, KL_h1, acc_h1, bern_MAS_h1, bern_2_h1, MASlocBern, locBern2, MASbeta, beta2, betaExp_MAS_h1, betaExp_2_h1, xlabel):
    '''Creates 4 subpanles for model estimation'''

    ax[0].set_xlabel(xlabel)
    #ax[0].set_xlim(10**(-100), 10**(10))
    ax[0].set_xscale('log')
    ax[0].set_ylim(0, 110)
    ax[0].errorbar(KL_h1, acc_h1, fmt='o', c='b', marker="*", fillstyle='none', label='Accuracy')
    ax[0].axhline(y=100, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='b')  # linestyle is dashdotdotted
    ax[0].set_ylabel('Accuracy (%)', rotation=90)
    ax[0].get_yticks()
    ax[0].set_title('Accuracy')
    ax[0].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.0e'))
