""" To draw transition matrix """
from .figure11 import lapt_tHMMobj_list
from .figure12 import gemc_tHMMobj_list
from .common import getSetup, subplotLabel
from ..plotTree import plot_networkx


gemc = gemc_tHMMobj_list[0]
lapt = lapt_tHMMobj_list[0]
T_lap = lapt_tHMMobj_list[0].estimate.T
T_gem = gemc_tHMMobj_list[0].estimate.T


def makeFigure():
    """ makes figure 13 for transition matrices. """

    ax, f = getSetup((13, 8), (1, 2))
    subplotLabel(ax)

    # transition matrix lapatinib
    plot_networkx(T_lap, "lpt")
    ax[0].axis("off")
    ax[0].set_title("lapatinib")

    # transition matrix
    plot_networkx(T_gem, "gmc")
    ax[1].axis("off")
    ax[1].set_title("gemcitabine")

    return f
