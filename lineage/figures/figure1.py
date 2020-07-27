"""
File: figure1.py
Purpose: Generates figure 1.
Figure 1 is the tHMM model and its purpose.
"""
import numpy as np

from .figureCommon import getSetup, subplotLabel
from ..LineageTree import LineageTree
from ..plotTree import plotLineage
from ..states.StateDistributionGaPhs import StateDistribution as phaseStateDist

# create emissions for homogeneous and heterogeneous lineages
state20 = phaseStateDist(1.0, 1.0, 8, 7, 4, 2)
state21 = phaseStateDist(1.0, 1.0, 16, 4, 3, 5)
E2 = [state20, state21]
pi = np.array([0.95, 0.05], dtype="float")

# create homogeneous population (1-state)
T1 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float")
HomoX = LineageTree.init_from_parameters(pi, T1, E2, desired_num_cells=2**7 - 1, censor_condition=0, desired_experiment_time=400)

# create heterogeneous population (2-state)
T2 = np.array([[0.9, 0.1], [0.0, 1.0]], dtype="float")
HeteroX = LineageTree.init_from_parameters(pi, T2, E2, desired_num_cells=2**7 - 1, censor_condition=0, desired_experiment_time=400)

def makeFigure():
    """
    Makes figure 1.
    """

    plotLineage(HomoX, 'lineage/figures/cartoons/figure1a.svg', censore=False)
    plotLineage(HeteroX, 'lineage/figures/cartoons/figure1b.svg', censore=False)
    # Get list of axis objects
    ax, f = getSetup((7, 10 / 3), (1, 2))
    figureMaker(ax)
    subplotLabel(ax)

    return f

def figureMaker(ax):
    """
    Makes figure 1.
    """
    i = 0
    ax[i].axis('off')
    i += 1
    ax[i].axis('off')
