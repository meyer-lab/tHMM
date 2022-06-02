""" Unit tests for crossvalidation. """
import pytest
from ..LineageTree import LineageTree
from ..figures.common import pi, T, E
from ..CrossVal import crossval, hide_for_population

@pytest.mark.parametrize("censored", [0, 3])
@pytest.mark.parametrize("num_cells", [7, 15])
def test_cv(censored, num_cells):
    complete_lineages = [[LineageTree.init_from_parameters(pi, T, E, num_cells, censor_condition=censored, desired_experiment_time=150) for _ in range(20)] for _ in range(4)]

    train_lineages, hidden_indexes, hidden_obs = hide_for_population(complete_lineages, 0.2)

    ll = []
    for i in range(1, 3):
        ll.append(crossval(train_lineages, hidden_indexes, hidden_obs, i))
    assert ll[0] < ll[1]
