""" Figure 20 for correlations between cell observations in the lineages. """

import numpy as np
import pandas as pd
from .common import getSetup
from ..Lineage_collections import AllLapatinib, AllGemcitabine


def makeFigure():
    """
    Makes figure 9.
    """
    ax, f = getSetup((9, 4), (1, 2))

    labels = ["G1", "SG2"]
    x = np.arange(len(labels))  # the label locations
    width = 0.35
    G1s = [0.515, 0.467]
    G2s = [0.368, 0.166]
    G3s = [1e-10, 3e-16]
    ax[0].bar(x - width / 2, G1s, width / 2, label="mother")
    ax[0].bar(x, G2s, width / 2, label="grandmother")
    ax[0].bar(x + width / 2, G3s, width / 2, label="great-grandmother")
    ax[0].set_title("lapatinib")
    ax[0].set_ylabel("surv. Spearman corr.")
    ax[0].set_xticks(x, labels)
    ax[0].legend()
    ax[0].set_ylim((-1, 1))

    G1sg = [0.238, 0.697]
    G2sg = [0.143, 0.026]
    G3sg = [-0.228, 6e-16]
    ax[1].bar(x - width / 2, G1sg, width / 2, label="mother")
    ax[1].bar(x, G2sg, width / 2, label="grandmother")
    ax[1].bar(x + width / 2, G3sg, width / 2, label="great-grandmother")
    ax[1].set_title("gemcitabine")
    ax[1].set_ylabel("surv. Spearman corr.")
    ax[1].set_xticks(x, labels)
    ax[1].legend()
    ax[1].set_ylim((-1, 1))

    return f


def save_df():
    """Save the arrays that are used for calculating the correlation into dataframes, in the form of column1:gen1, column2: gen2.
    This functions does this for gen1 & 2, gen 1 & 3, gen 1 & 4, and for both G1 and S-G2 cell lifetimes.
    """

    cells_l, mothers_l, cells_censored_l, mothers_censored_l = [], [], [], []
    for population in AllLapatinib:
        cs, ms, cs_censored, ms_censored = get_obs_population(population, 3)
        cells_l += cs
        mothers_l += ms
        cells_censored_l += cs_censored
        mothers_censored_l += ms_censored

    cells_g, mothers_g, cells_censored_g, mothers_censored_g = [], [], [], []
    for population in AllGemcitabine:
        cs, ms, cs_censored, ms_censored = get_obs_population(population, 3)
        cells_g += cs
        mothers_g += ms
        cells_censored_g += cs_censored
        mothers_censored_g += ms_censored

    df1 = pd.DataFrame(
        {
            "cells": cells_l,
            "mothers": mothers_l,
            "cells_censor": cells_censored_l,
            "mother_censor": mothers_censored_l,
        }
    )
    df2 = pd.DataFrame(
        {
            "cells": cells_g,
            "mothers": mothers_g,
            "cells_censor": cells_censored_g,
            "mother_censor": mothers_censored_g,
        }
    )

    df1.to_csv(r"lap_g2.csv", index=False)
    df2.to_csv(r"gem_g2.csv", index=False)


def get_obs_lineage(lineage, G1sG2: int):
    """G1sG2 is either 2: G1, or 3: SG2. We start from second cells to avoid appending root cells in cells."""
    cells, mothers, cells_censored, mothers_censored = [], [], [], []
    for cell in lineage.output_lineage[3:]:
        if cell.parent.parent.parent:
            cells.append(cell.obs[G1sG2])
            mothers.append(cell.parent.parent.parent.obs[G1sG2])
            cells_censored.append(cell.obs[G1sG2 + 2])
            mothers_censored.append(cell.parent.parent.parent.obs[G1sG2 + 2])

    return cells, mothers, cells_censored, mothers_censored


def get_obs_population(population, i):
    """Given a list of lineages it creates lists of observations and censorship for cells with 1 generation difference."""
    cells, mothers, cells_censored, mothers_censored = [], [], [], []
    for lineage in population:
        cs, ms, cs_censored, ms_censored = get_obs_lineage(lineage, i)
        cells += cs
        mothers += ms
        cells_censored += cs_censored
        mothers_censored += ms_censored

    return cells, mothers, cells_censored, mothers_censored


# R code to calculate the survSpearman:
# library(survSpearman)
# data <- read.csv("data.csv")
# dt <- na.omit(data)
# corr <- survSpearman(dt[,1], dt[,2], dt[,3], dt[,4])$Correlation[1] # this gives the highest rank correlation

# correlations:
# gem_g1: 0.2382719
# gem_g2: 0.6971354
# gem_g1_grand: 0.1431158
# gem_g2_grand: 0.0263866

# lap_g1: 0.5153418
# lap_g2: 0.4675189
# lap_g1_grand: 0.368836
# lap_g2_grand: 0.1661138
