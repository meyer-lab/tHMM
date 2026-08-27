"""Test cross validation."""

import numpy as np
import pytest

from ..crossval import crossval, hide_observation
from ..figures.common import E2, T, pi
from ..LineageTree import LineageTree


def test_hide_obs():
    """Test that we are correctly hiding observations."""
    complete_lineages = [LineageTree.rand_init(pi, T, E2, 31) for _ in range(10)]

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
    it checks that the log-likelihood of a 2 state model is higher than a 1 state model.
    """
    local_rng = np.random.default_rng(cen + 5)
    complete_lineages = [
        [
            LineageTree.rand_init(pi, T, E2, 31, censor_condition=cen, desired_experiment_time=150, rng=local_rng)
            for _ in range(20)
        ]
        for _ in range(4)
    ]

    train_lineages = [hide_observation(complete_lin, 0.25, rng=local_rng) for complete_lin in complete_lineages]

    ll = crossval(train_lineages, np.arange(1, 3), rng=local_rng)
    assert ll[0] < ll[1]
