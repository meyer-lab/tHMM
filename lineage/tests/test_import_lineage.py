""" Unit test for the new AU565 data. """

import numpy as np

from ..CellVar import CellVar
from ..import_lineage import import_AU565, MCF10A


def test_data():
    """Manually setting up the first two lineages from the new AU565 data."""
    cell1 = CellVar(parent=None)
    cell1.left = CellVar(parent=cell1)
    cell1.right = CellVar(parent=cell1)

    cell1.obs = [1, 3.0, 11.32, 1]
    cell1.left.obs = [0, 6.0, 11.83, 0]
    cell1.right.obs = [0, 1.5, 1.41, 0]
    cell2 = cell1.left
    cell3 = cell1.right

    lin1 = [cell1, cell2, cell3]

    cell4 = CellVar(parent=None)
    cell4.obs = [1, 14.5, 13.21, 1]
    cell4.left = CellVar(parent=cell4)
    cell4.left.obs = [np.nan, 9.5, 11.15, 1]
    cell4.right = CellVar(parent=cell4)
    cell4.right.obs = [np.nan, 9.5, 11.02, 1]
    lin2 = [cell4, cell4.left, cell4.right]

    lineages = import_AU565(
        "lineage/data/LineageData/AU02101_A3_field_1_RP_50_CSV-Table.csv"
    )
    lin1 = lineages[0]  # lineage ID = 2
    lin2 = lineages[2]  # lineage ID = 3

    assert len(lin1) == 3
    assert len(lin2) == 3

    for i, cell in enumerate(lin1):
        np.testing.assert_allclose(cell.obs, lin1[i].obs, rtol=1e-2)

    for j, cells in enumerate(lin2):
        np.testing.assert_allclose(cells.obs, lin2[j].obs, rtol=1e-2)


def test_MCF10A():
    pbs = MCF10A("PBS")
    egf = MCF10A("EGF")
    hgf = MCF10A("HGF")
    osm = MCF10A("OSM")
    # test for PBS
    lin1 = pbs[0]
    assert len(lin1) == 3  # has 3 cells
    np.testing.assert_allclose(lin1[0].obs, [1, 30.0, 0, 8.70, 4.35], rtol=1e-2)
    np.testing.assert_allclose(lin1[1].obs, [np.nan, 17.5, 0, 2.85, 1.42], rtol=1e-2)
    np.testing.assert_allclose(lin1[2].obs, [np.nan, 17.5, 0, 2.96, 1.48], rtol=1e-2)
