'''Generates a plot with 4 subplots for accuracy estimation and estimation of a bernoulli and 2 gompertz parameters. Note: this is only used for the Lineage Length and Lineage Number figures, the AIC only requires one figure of accuracy so it has its own maplotlib code in the Fig_Gen function.'''

from matplotlib import pyplot as plt
import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')


def Matplot_gen(ax, x, acc_h1, bern_MAS_h1, bern_2_h1, MASlocBern, locBern2, MASbeta, beta2, betaExp_MAS_h1, betaExp_2_h1, xlabel, FOM='E'):
    '''Creates 4 subpanels for model estimation'''
    print("length of ax: {}".format(len(ax)))
    font = 11
    font2 = 10

    if FOM == 'E':
        panel_3_title = 'Exponential'
        panel_3_ylabel = 'Labmda'

    #fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
    #ax = axs[0, 0]
    ax[0].set_xlim((0, int(round(1.1 * max(x)))))
    ax[0].set_xlabel(xlabel, fontsize=font2)
    ax[0].set_ylim(0, 110)
    ax[0].errorbar(x, acc_h1, fmt='o', c='k', marker="o", label='Accuracy', alpha=0.5)
    ax[0].axhline(y=100, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='k')  # linestyle is dashdotdotted
    ax[0].set_ylabel('Accuracy (%)', rotation=90, fontsize=font2)
    ax[0].get_yticks()
    ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[0].get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0].set_title('Accuracy', fontsize=font)
    ax[0].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.4)

    #ax = axs[0, 1]
    ax[1].set_xlim((0, int(round(1.1 * max(x)))))
    ax[1].set_xlabel(xlabel, fontsize=font2)
    ax[1].errorbar(x, bern_MAS_h1, fmt='o', c='b', marker="o", label='State 1', alpha=0.5)
    ax[1].errorbar(x, bern_2_h1, fmt='o', c='r', marker="o", label='State 2', alpha=0.5)
    ax[1].set_ylabel('Theta', rotation=90, fontsize=font2)
    ax[1].axhline(y=MASlocBern, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b')
    ax[1].axhline(y=locBern2, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r')
    ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[1].get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[1].set_title('Bernoulli', fontsize=font)
    ax[1].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.4)

    #ax = axs[1, 0]
    ax[2].set_xlim((0, int(round(1.1 * max(x)))))
    ax[2].set_xlabel(xlabel, fontsize=font2)
    #ax[2].set_xscale("log", nonposx='clip')
    ax[2].errorbar(x, betaExp_MAS_h1, fmt='o', c='b', marker="o", label='State 1', alpha=0.5)
    ax[2].errorbar(x, betaExp_2_h1, fmt='o', c='r', marker="o", label='State 2', alpha=0.75)
    ax[2].axhline(y=MASbeta, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b')
    ax[2].axhline(y=beta2, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r')
    ax[2].set_ylabel(panel_3_ylabel, rotation=90, fontsize=font2)
    ax[2].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[2].get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[2].set_title(panel_3_title, fontsize=font)
    ax[2].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.4)
