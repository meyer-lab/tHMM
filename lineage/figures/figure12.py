""" This file is only to plot the distribution of death lengths. """
import numpy as np
import seaborn as sns
import networkx as nx

from .figure11 import gemc_tHMMobj_list, lapt_tHMMobj_list, concs
from .figureCommon import getSetup, subplotLabel

def makeFigure():
    """ Makes figure 11. """

    ax, f = getSetup((13.2, 6.5), (2, 4))
    subplotLabel(ax)

    for i in range(4):
        gemc = gemc_tHMMobj_list[i]
        lapt = lapt_tHMMobj_list[i]
        gm = []
        lp = []
        for lin in gemc.X:
            for cell in lin.output_lineage:
                if cell.hasDied():
                    if np.isfinite(cell.obs[2]) and np.isfinite(cell.obs[3]):
                        length = cell.obs[2] + cell.obs[3]
                    elif np.isnan(cell.obs[2]):
                        length = cell.obs[3]
                    elif np.isnan(cell.obs[3]):
                        length = cell.obs[2]
                    gm.append(length)

        for lin in lapt.X:
            for cell in lin.output_lineage:
                if cell.hasDied():
                    if np.isfinite(cell.obs[2]) and np.isfinite(cell.obs[3]):
                        length = cell.obs[2] + cell.obs[3]
                    elif np.isnan(cell.obs[2]):
                        length = cell.obs[3]
                    elif np.isnan(cell.obs[3]):
                        length = cell.obs[2]
                    lp.append(length)

        ax[i].hist(lp, bins=30)
        ax[i+4].hist(gm, bins=30)
        ax[i].set_ylabel("freq.")
        ax[i].set_xlabel("lived before death")
        ax[i+4].set_ylabel("freq.")
        ax[i+4].set_xlabel("lived before death")

        ax[i].set_title("lapatinib "+str(concs[i]))
        ax[i+4].set_title("gemcitabine"+str(concs[i+4]))

    return f
