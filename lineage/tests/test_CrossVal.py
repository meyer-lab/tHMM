""" Test cross validation. """
from ..LineageTree import LineageTree
from ..figures.common import pi, T, E2
from ..crossval import hide_observation


def test_hide_obs():
    """Test that we are correctly hiding observations."""
    complete_lineages = [LineageTree.init_from_parameters(pi, T, E2, 31, censor_condition=0, desired_experiment_time=150) for _ in range(10)]

    train_lineages = hide_observation(complete_lineages, 0.25)

    negatives, total = 0, 0
    for lin in train_lineages:
        for cell in lin.output_lineage:
            total += 1
            if (cell.obs[2] < 0.0):
                negatives += 1

    assert 0.2 <= negatives / total <= 0.3
