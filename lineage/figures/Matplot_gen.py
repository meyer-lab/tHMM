'''Generates a plot with 4 subplots for accuracy estimation and estimation of a bernoulli and 2 gompertz parameters. Note: this is only used for the Lineage Length and Lineage Number figures, the AIC only requires one figure of accuracy so it has its own maplotlib code in the Fig_Gen function.'''

from matplotlib import pyplot as plt
import matplotlib
import matplotlib.ticker
import numpy as np
matplotlib.use('Agg')


def moving_average(a, n=10):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def Matplot_gen(ax, x, acc_h1, bern_MAS_h1, bern_2_h1, MASlocBern, locBern2, MASbeta, beta2, betaExp_MAS_h1, betaExp_2_h1, xlabel, FOM='E'):
    '''Creates 4 subpanels for model estimation'''
    print("length of ax: {}".format(len(ax)))
    font = 11
    font2 = 10

    if FOM == 'E':
        panel_3_title = 'Exponential'
        panel_3_ylabel = 'Labmda'

    x_vs_acc = np.column_stack((x, acc_h1))
    sorted_x_vs_acc = x_vs_acc[np.argsort(x_vs_acc[:, 0])]

    #fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
    #ax = axs[0, 0]
    ax[0].set_xlim((0, int(round(1.1 * max(x)))))
    ax[0].set_xlabel(xlabel, fontsize=font2)
    ax[0].set_ylim(0, 110)
    ax[0].errorbar(x, acc_h1, fmt='o', c='k', marker="o", label='Accuracy', alpha=0.3)
    ax[0].plot(sorted_x_vs_acc[:, 0][9:], moving_average(sorted_x_vs_acc[:, 1]), c='k', label='Moving Average')
    ax[0].axhline(y=100, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='k', alpha=0.6)  # linestyle is dashdotdotted
    ax[0].set_ylabel('Accuracy (%)', rotation=90, fontsize=font2)
    ax[0].get_yticks()
    ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[0].get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.4)
    ax[0].set_title('Accuracy', fontsize=font)

    #ax = axs[0, 1]
    ax[1].set_xlim((0, int(round(1.1 * max(x)))))
    ax[1].set_xlabel(xlabel, fontsize=font2)
    ax[1].errorbar(x, bern_MAS_h1, fmt='o', c='b', marker="o", label='Susceptible', alpha=0.2)
    ax[1].errorbar(x, bern_2_h1, fmt='o', c='r', marker="o", label='Resistant', alpha=0.2)
    ax[1].set_ylabel('Theta', rotation=90, fontsize=font2)
    ax[1].axhline(y=MASlocBern, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b', alpha=0.6)
    ax[1].axhline(y=locBern2, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[1].get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[1].set_title('Bernoulli', fontsize=font)
    ax[1].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.3)
    ax[1].legend(loc='best', framealpha=0.3)

    #ax = axs[1, 0]
    ax[2].set_xlim((0, int(round(1.1 * max(x)))))
    ax[2].set_xlabel(xlabel, fontsize=font2)
    #ax[2].set_xscale("log", nonposx='clip')
    ax[2].errorbar(x, betaExp_MAS_h1, fmt='o', c='b', marker="o", label='Susceptible', alpha=0.2)
    ax[2].errorbar(x, betaExp_2_h1, fmt='o', c='r', marker="o", label='Resistant', alpha=0.2)
    ax[2].axhline(y=MASbeta, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='b', alpha=0.6)
    ax[2].axhline(y=beta2, linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2, color='r', alpha=0.6)
    ax[2].set_ylabel(panel_3_ylabel, rotation=90, fontsize=font2)
    ax[2].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[2].get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[2].set_title(panel_3_title, fontsize=font)
    ax[2].tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.3)
    ax[2].legend(loc='best', framealpha=0.3)
