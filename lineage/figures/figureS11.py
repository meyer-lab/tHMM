""" This is a file to put together 4 conditions of lapatinib together. """

from .figureCommon import getSetup

from string import ascii_lowercase


def makeFigure():
    """
    Makes figure 100.
    """
    titles = ["Control", "Lapatinib 25 nM", "Lapatinib 50 nM", "Lapatinib 250 nM"]
    ax, f = getSetup((10, 11), (1, 4))

    for i in range(4):
        ax[i].axis('off')
        ax[i].text(-0.2, 1.25, ascii_lowercase[i], transform=ax[i].transAxes, fontsize=16, fontweight="bold", va="top")
        ax[i].text(0.03, 1.25, titles[i], transform=ax[i].transAxes, fontsize=12, va="top")

    return f
