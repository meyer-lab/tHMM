'''Generates a plot with 4 subplots for accuracy estimation and estimation of a bernoulli and 2 gompertz parameters. Note: this is only used for the Lineage Length and Lineage Number figures, the AIC only requires one figure of accuracy so it has its own maplotlib code in the Fig_Gen function.'''

from matplotlib import pyplot as plt
import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')


def Matplot_gen(ax, x, acc_h1, bern_MAS_h1, bern_2_h1, MASlocBern, locBern2, betaExp_MAS_h1, betaExp_2_h1, MASbetaExp, betaExp2, xlabel):
    '''Creates 4 subpanles for model estimation'''

    #fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
    #ax = axs[0, 0]
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylim(0, 110)
    ax[0].errorbar(x, acc_h1, fmt='o', c='b', marker="*", fillstyle='none', label='Accuracy')
    ax[0].axhline(y=100, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='b')  # linestyle is dashdotdotted
    ax[0].set_ylabel('Accuracy (%)', rotation=90)
    ax[0].get_yticks()
    ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[0].get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0].set_title('Accuracy')

    #ax = axs[0, 1]
    ax[1].set_xlabel(xlabel)
    ax[1].errorbar(x, bern_MAS_h1, fmt='o', c='g', marker="^", fillstyle='none', label='State 1')
    ax[1].errorbar(x, bern_2_h1, fmt='o', c='r', marker="^", fillstyle='none', label='State 2')
    ax[1].set_ylabel('Bernoulli', rotation=90)
    ax[1].axhline(y=MASlocBern, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='g')
    ax[1].axhline(y=locBern2, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='r')
    ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[1].get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[1].set_title('Bernoulli')

    #ax = axs[1, 1]
    ax[2].set_xlabel(xlabel)
    #ax[2].set_xscale("log", nonposx='clip')
    ax[2].errorbar(x, betaExp_MAS_h1, fmt='o', c='g', marker="^", fillstyle='none', label='State 1')
    ax[2].errorbar(x, betaExp_2_h1, fmt='o', c='r', marker="^", fillstyle='none', label='State 2')
    ax[2].axhline(y=MASbetaExp, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='g')
    ax[2].axhline(y=betaExp2, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='r')
    ax[2].set_ylabel('Exponential Beta', rotation=90)
    ax[2].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[2].get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[2].set_title('Exponential')
