""" This is a file to put together 4 conditions of lapatinib together. """

from .common import getSetup

from string import ascii_lowercase


def makeFigure():
    """
    Makes figure 170.
    """
    titles = ["Control gemcitabine LAP", "Control gemcitabine GEM"]
    ax, f = getSetup((7, 40), (1, 2))

    for i in range(2):
        ax[i].axis("off")
        ax[i].text(
            -0.15,
            1.25,
            ascii_lowercase[i],
            transform=ax[i].transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
        )
        ax[i].text(
            0.0, 1.25, titles[i], transform=ax[i].transAxes, fontsize=11, va="top"
        )

    return f
