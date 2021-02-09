""" This is a file to put together 4 conditions of lapatinib together. """

from .figureCommon import getSetup

from string import ascii_lowercase


def makeFigure():
    """
    Makes figure 100.
    """
    titles = ["Control", "25 nM", "50 nM", "250 nM"]
    ax, f = getSetup((10, 7), (1, 4))

    for i in range(4):
        ax[i].axis('off')
        ax[i].text(-0.2, 1.25, ascii_lowercase[i], transform=ax[i].transAxes, fontsize=16, fontweight="bold", va="top")
        ax[i].text(0.3, 1.25, titles[i], transform=ax[i].transAxes, fontsize=16, va="top")

    return f
