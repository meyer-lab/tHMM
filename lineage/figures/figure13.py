""" This file is only to plot the distribution of death lengths. """
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns

from .figure11 import lapt_tHMMobj_list, concs
from .figure12 import gemc_tHMMobj_list
from .figureCommon import getSetup, subplotLabel


def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf

lp_pdf = []
gm_pdf = []
lp_params = [(11.0, 49.59), (14.0, 21.66), (3.0, 42.66), (8.0, 60.02)]
gm_params = [(11.0, 49.59), (13.0, 41.43), (1.0, 76.82), (1.0, 80.78)]
for i in range(4):
    best_dist = getattr(st, 'expon')
    lp_pdf.append(make_pdf(best_dist, lp_params[i]))
    gm_pdf.append(make_pdf(best_dist, gm_params[i]))

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
                if cell.obs[0] == 0 or cell.obs[1] == 0:
                    if np.isfinite(cell.obs[2]) and np.isfinite(cell.obs[3]):
                        length = cell.obs[2] + cell.obs[3]
                    elif np.isnan(cell.obs[2]):
                        length = cell.obs[3]
                    elif np.isnan(cell.obs[3]):
                        length = cell.obs[2]
                    gm.append(length)

        for lin in lapt.X:
            for cell in lin.output_lineage:
                if cell.obs[0] == 0 or cell.obs[1] == 0:
                    if np.isfinite(cell.obs[2]) and np.isfinite(cell.obs[3]):
                        length = cell.obs[2] + cell.obs[3]
                    elif np.isnan(cell.obs[2]):
                        length = cell.obs[3]
                    elif np.isnan(cell.obs[3]):
                        length = cell.obs[2]
                    lp.append(length)

        ax[i].hist(lp, density=True, bins=30)
        ax[i].plot(lp_pdf[i])
        ax[i + 4].hist(gm, density=True, bins=30)
        ax[i + 4].plot(gm_pdf[i])
        ax[i].set_ylabel("freq.")
        ax[i].set_xlabel("lived before death")
        ax[i + 4].set_ylabel("freq.")
        ax[i + 4].set_xlabel("lived before death")
        ax[i].set_title("lapatinib " + str(concs[i]))
        ax[i + 4].set_title("gemcitabine" + str(concs[i + 4]))

    return f
