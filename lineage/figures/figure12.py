""" This file depicts the distribution of phase lengths versus the states. """
from matplotlib.ticker import MaxNLocator

from .figureCommon import getSetup
from .figure11 import lapatinib12, gemcitabine12


def makeFigure():
    """
    Makes figure 12.
    """
    ax, f = getSetup((7, 6), (2, 4))
    # titles
    concs = ["cntrl", "Lapt 25uM", "Lapt 50 uM", "Lapt 250 uM", "cntrl", "Gem 5uM", "Gem 10uM", "Gem 30uM"]

    # plot
    for i in range(4):
        ax[i].scatter([a[0] for a in lapatinib12[i]], [a[1] for a in lapatinib12[i]], alpha=0.3, marker="+", c="#00ffff")
        ax[i].set_ylabel("G1 phase lengths")
        ax[i].set_xlabel("state")
        ax[i].set_title(concs[i])
        ax[i].set_ylim(bottom=-20, top=180)
        ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))

    for i in range(4, 8):
        ax[i].scatter([a[0] for a in gemcitabine12[i-4]], [a[2] for a in gemcitabine12[i-4]], alpha=0.3, marker="+", c="#feba4f")
        ax[i].set_ylabel("G2 phase lengths")
        ax[i].set_xlabel("state")
        ax[i].set_title(concs[i])
        ax[i].set_ylim(bottom=-20, top=180)
        ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))

    return f
