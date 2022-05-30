""" Unit tests for crossvalidation. """
import pytest
from ..LineageTree import LineageTree
from ..figures.common import pi, T, E
from ..CrossVal import crossval, hide_observation

@pytest.mark.parametrize("censored", [0, 3])
def test_cv(censored):
    complete_lineages = [LineageTree.init_from_parameters(pi, T, E, 15, censor_condition=censored, desired_experiment_time=150) for _ in range(100)]
    print(len(complete_lineages[0].output_lineage))
    train_lineages, hidden_indexes, hidden_obs = [], [], []
    for complete_lin in complete_lineages:
        lineage, hide_index, hide_obs = hide_observation(complete_lin, 0.3)
        train_lineages.append(lineage)
        hidden_indexes.append(hide_index)
        hidden_obs.append(hide_obs)

    ll = []
    for i in range(1, 3):
        ll.append(crossval(train_lineages, hidden_indexes, hidden_obs, i))
    assert ll[0] < ll[1]
