'''Generates a plot with 4 subplots for accuracy estimation and estimation of a bernoulli and 2 gompertz parameters. Note: this is only used for the Lineage Length and Lineage Number figures, the AIC only requires one figure of accuracy so it has its own maplotlib code  in the Fig_Gen function.'''

import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.ticker

def Matplot_gen(x,acc_h1,bern_MAS_h1,bern_2_h1,MASlocBern,locBern2,cGom_MAS_h1,cGom_2_h1,MAScGom,
                cGom2,scaleGom_MAS_h1,scaleGom_2_h1,MASscaleGom,scaleGom2, xlabel, title, save_name):
    
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
    ax = axs[0,0]
    ax.set_ylim(0,110)
    l1 = ax.errorbar(x, acc_h1, fmt='o', c='b',marker="*",fillstyle='none', label = 'Accuracy')
    ax.axhline(y=100, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='b',)
    ax.set_ylabel('Accuracy (%)',rotation=90)
    vals = ax.get_yticks()

    ax = axs[0,1]
    l2 = ax.errorbar(x, bern_MAS_h1, fmt='o', c='g',marker="^",fillstyle='none', label = 'State 1')
    l3 = ax.errorbar(x, bern_2_h1, fmt='o', c='r',marker="^",fillstyle='none', label = 'State 2')
    ax.set_ylabel('Bernoulli', rotation=90)
    ax.axhline(y= MASlocBern, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='g')
    ax.axhline(y=locBern2, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='r')

    ax = axs[1,0]
    ax.set_xlabel(xlabel)
    ax.set_xscale("log", nonposx='clip')
    ax.errorbar(x,cGom_MAS_h1, fmt='o',c='g',marker="^",fillstyle='none', label = 'State 1')
    ax.errorbar(x,cGom_2_h1, fmt='o',c='r',marker="^",fillstyle='none', label = 'State 2')
    ax.axhline(y=MAScGom, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='g')
    ax.axhline(y=cGom2, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='r')
    ax.set_ylabel('Gompertz C',rotation=90)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())


    ax = axs[1,1]
    ax.set_xlabel(xlabel)
    ax.set_xscale("log", nonposx='clip')
    ax.errorbar(x,scaleGom_MAS_h1, fmt='o',c='g',marker="^",fillstyle='none', label = 'State 1')
    ax.errorbar(x,scaleGom_2_h1, fmt='o',c='r',marker="^",fillstyle='none', label = 'State 2')
    ax.axhline(y=MASscaleGom, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='g')
    ax.axhline(y=scaleGom2, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='r')
    ax.set_ylabel('Gompertz Scale',rotation=90)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(title)
    fig.savefig(save_name)
