""" Unit tests for crossvalidation. """
from ..LineageTree import LineageTree
from ..figures.common import pi, T, E
from ..figures.figure19 import crossval, hide_observation


def test_cv():
    complete_lineages = [LineageTree.init_from_parameters(pi, T, E, 15) for _ in range(5)]
    train_lineages, hidden_indexes, hidden_obs = [], [], []
    for complete_lin in complete_lineages:
        lineage, hide_index, hide_obs = hide_observation(complete_lin, 0.2)
        train_lineages.append(lineage)
        hidden_indexes.append(hide_index)
        hidden_obs.append(hide_obs)

    ll = []
    for i in range(1, 3):
        ll.append(crossval(train_lineages, hidden_indexes, hidden_obs, i))
    assert ll[0] < ll[1]