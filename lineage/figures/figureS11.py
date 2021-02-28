""" This is a file to put together 4 conditions of lapatinib together. """

from .figureCommon import getSetup

from string import ascii_lowercase


def makeFigure():
    """
    Makes figure 100.
    """
    titles = ["Control", "lapatinib 25 nM", "lapatinib 50 nM", "lapatinib 250 nM"]
    ax, f = getSetup((10, 17), (1, 4))

    for i in range(4):
        ax[i].axis('off')
        ax[i].text(-0.2, 1.25, ascii_lowercase[i], transform=ax[i].transAxes, fontsize=16, fontweight="bold", va="top")
        ax[i].text(0.03, 1.25, titles[i], transform=ax[i].transAxes, fontsize=12, va="top")

    return f