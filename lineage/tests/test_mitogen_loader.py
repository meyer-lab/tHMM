"""Tests for the S-BSST314 (Min et al. 2020) MEK/ERK inhibitor pulse loader."""

import numpy as np
import pandas as pd
import pytest
import scipy.io as sio

from ..mitogen_analysis import relative_correlations
from ..mitogen_loader import (
    CELL_TABLE,
    CYTO_COL,
    DRUG_FRAME,
    N_FRAMES,
    NUC_COL,
    PULSES_H,
    build_cell_table,
    generation_depths,
    load_lineages,
    well_condition,
)
from ..palbociclib_loader import FRAME_HOURS


def test_well_condition_matches_plate_map():
    assert well_condition(2, 2) == ("MEKi", 1.0)
    assert well_condition(7, 5) == ("MEKi", 9.0)
    assert well_condition(4, 6) == ("ERKi", 1.0)
    assert well_condition(2, 9) == ("ERKi", 9.0)
    assert well_condition(3, 10) == ("MEKi", np.inf)
    assert well_condition(4, 10) == ("ERKi", np.inf)
    assert well_condition(5, 11) == ("none", 0.0)
    with pytest.raises(AssertionError):
        well_condition(1, 2)


@pytest.fixture
def movie_dir(tmp_path):
    """A MEKi 3 h movie (column 3) and a control movie (column 11), each with one root
    born before the drug that divides after it into two daughters, one of which divides."""
    d = DRUG_FRAME
    tracks = {
        0: (d - 50, d + 9),
        1: (d + 10, d + 79),
        2: (d + 10, N_FRAMES - 1),
        3: (d + 80, d + 90),
        4: (d + 80, d + 85),
        5: (0, d - 51),  # the root's mother, so that the root's birth is seen
    }
    mothers = [5, 0, 0, 1, 1, None]
    for col in (3, 11):
        trace = np.full((6, N_FRAMES, 14), np.nan)
        for i, (a, b) in tracks.items():
            trace[i, a : b + 1, NUC_COL] = 1000.0
            trace[i, a : b + 1, CYTO_COL] = 500.0 + 10.0 * np.arange(b - a + 1)
        gen = np.array([np.nan if m is None else m + 1 for m in mothers], dtype=float)[:, None]
        sio.savemat(tmp_path / f"tracedata_2_{col}_1.mat", {"tracedata": trace, "genealogy": gen, "jitters": 0})
    return tmp_path


def test_build_and_load(movie_dir):
    table = build_cell_table(str(movie_dir))
    assert set(zip(table["drug"], table["pulse_h"], strict=True)) == {("MEKi", 3.0), ("none", 0.0)}

    [lin] = load_lineages("MEKi", 3.0, table)
    [ctrl] = load_lineages("anything", 0.0, table)
    np.testing.assert_array_equal(lin.obs, ctrl.obs)
    assert len(lin) == 5
    np.testing.assert_array_equal(generation_depths([lin]), [0, 1, 1, 2, 2])
    # The root is truncated to dividing between the drug and the last frame.
    np.testing.assert_allclose(lin.obs[0, 1], 60 * FRAME_HOURS)
    np.testing.assert_allclose(lin.obs[0, 3:5], np.array([50, N_FRAMES - 1 - (DRUG_FRAME - 50)]) * FRAME_HOURS)
    assert lin.obs[1, 2] == 1.0 and lin.obs[2, 2] == 0.0
    assert np.isfinite(lin.obs[1, 0])  # G1 CDK2 activation rate


def test_relative_correlations(movie_dir):
    table = build_cell_table(str(movie_dir))
    rel = relative_correlations(load_lineages("MEKi", 3.0, table))
    # One mother-daughter pair (1 -> 3, 4; the root's G1 is pre-drug), two sister pairs,
    # and no cousins (only one of the sisters divided).
    assert rel["mother_daughter"]["n"] <= 2
    assert rel["sisters"]["n"] <= 2
    assert rel["cousins"]["n"] == 0


def test_shipped_table_loads():
    table = pd.read_csv(CELL_TABLE)
    assert set(table["pulse_h"]) == set(PULSES_H)
    table = table[table["movie"].str.endswith("_1")]  # one site per well keeps this quick
    for pulse in (0.0, np.inf):
        lins = load_lineages("MEKi", pulse, table)
        assert len(lins) > 100
        obs = np.vstack([lin.obs for lin in lins[:200]])
        born = np.isfinite(obs[:, 1])
        assert np.all(obs[born, 1] > 0)
        roots = np.concatenate([np.arange(len(lin)) == 0 for lin in lins[:200]])
        trunc = np.isfinite(obs[:, 3])
        # Only roots seen from birth are truncated, and each lies in its window.
        assert np.all(roots[trunc]) and np.all(born[trunc])
        assert np.all((obs[trunc, 3] <= obs[trunc, 1]) & (obs[trunc, 1] <= obs[trunc, 4]))
