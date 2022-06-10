""" Test cross validation. """
import numpy as np
import pytest
from ..LineageTree import LineageTree
from ..figures.common import pi, T, E2
from ..crossval import hide_observation, crossval
from ..Lineage_collections import Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM


def test_hide_obs():
    """Test that we are correctly hiding observations."""
    complete_lineages = [LineageTree.init_from_parameters(pi, T, E2, 31) for _ in range(10)]

    train_lineages = hide_observation(complete_lineages, 0.25)

    negatives, total = 0, 0
    for lin in train_lineages:
        for cell in lin.output_lineage:
            total += 1
            if cell.obs[2] < 0.0:
                negatives += 1

    assert 0.2 <= negatives / total <= 0.3


@pytest.mark.parametrize("cen", [0, 3])
def test_cv(cen):
    """For censored and uncensored 2-state synthetic data,
    it checks that the log-likelihood of a 2 state model is higher than a 1 state model."""
    complete_lineages = [
        [LineageTree.init_from_parameters(pi, T, E2, 7, censored_condition=cen, desired_experiment_time=100) for _ in range(50)] for _ in range(4)
    ]

    train_lineages = [hide_observation(complete_lin, 0.25) for complete_lin in complete_lineages]

    dataFull = []
    for _ in range(1, 3):
        dataFull.append(train_lineages)

    ll = crossval(dataFull, np.arange(1, 3))
    assert ll[0] < ll[1]
