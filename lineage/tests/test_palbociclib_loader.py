"""Tests for the Spencer-lab palbociclib live-imaging loader."""

import numpy as np
import pandas as pd
import pytest
import scipy.io as sio

from ..palbociclib_loader import (
    CYTO_COL,
    DRUG_FRAME,
    FRAME_HOURS,
    N_FRAMES,
    NUC_COL,
    build_cell_table,
    cell_obs,
    load_lineages,
    max_activation_rate,
    s_phase_entry,
)


def test_max_activation_rate_on_a_ramp():
    """A linear G1 ramp of 0.1 per hour is recovered, and the rate stops at S-phase entry."""
    hours = np.arange(80) * FRAME_HOURS
    cdk2 = 0.4 + 0.1 * hours
    cdk2[35:] = np.linspace(cdk2[34], 3.0, 45)  # a much steeper rise after S-phase entry
    assert max_activation_rate(cdk2) == pytest.approx(0.1, rel=1e-6)

    flat = np.full(80, 0.45)
    assert max_activation_rate(flat) == pytest.approx(0.0)
    assert np.isnan(max_activation_rate(flat[:12]))  # too short a G1 to measure


def test_s_phase_entry():
    cdk2 = np.full(100, 0.5)
    cdk2[40:] = 1.3
    # The 5-frame running median crosses at the step itself.
    assert s_phase_entry(cdk2) == pytest.approx(40 * FRAME_HOURS)
    assert np.isnan(s_phase_entry(np.full(100, 0.5)))


def write_movie(path, tracks, mothers):
    """Write a tracedata file: ``tracks`` maps cell index to (first, last, cdk2 level)."""
    n = len(mothers)
    trace = np.full((n, N_FRAMES, 9), np.nan)
    for i, (a, b, level) in tracks.items():
        trace[i, a : b + 1, NUC_COL] = 1000.0
        trace[i, a : b + 1, CYTO_COL] = 1000.0 * level
    genealogy = np.array([np.nan if m is None else m + 1 for m in mothers], dtype=float)[:, None]
    sio.savemat(path, {"tracedata": trace, "genealogy": genealogy, "jitters": np.zeros((N_FRAMES, 2))})


@pytest.fixture
def movie_dir(tmp_path):
    """One palbociclib movie (row 3) and one control movie (row 6).

    Cell 0 is born before the drug and divides after it into 1 and 2; cell 1 divides
    into 3 and 4 before the end; cell 2 is lost early; 3 runs to the end; 4 is lost.
    Cell 5 is a pre-drug cell with no mother and no post-drug daughters.
    """
    d = DRUG_FRAME
    tracks = {
        0: (d - 60, d + 9, 0.5),
        1: (d + 10, d + 69, 0.6),
        2: (d + 10, d + 30, 0.45),
        3: (d + 70, N_FRAMES - 1, 0.7),
        4: (d + 70, d + 80, 0.7),
        5: (0, 100, 0.4),
        6: (0, d - 61, 0.5),
    }
    mothers = [6, 0, 0, 1, 1, None, None]
    for row in (3, 6):
        write_movie(tmp_path / f"tracedata_{row}_1_1.mat", tracks, mothers)
    return tmp_path


def test_build_cell_table(movie_dir):
    table = build_cell_table(str(movie_dir))
    assert set(table["condition"]) == {"palbociclib", "control"}
    assert set(table.loc[table["condition"] == "palbociclib", "dose_nM"]) == {1000.0}
    one = table[table["movie"] == "3_1_1"].set_index("cell")
    # Post-drug cells (ids 2-5 in 1-based numbering) and their pre-drug mother (id 1).
    assert sorted(one.index) == [1, 2, 3, 4, 5]
    assert one.loc[1, "divided"] and one.loc[2, "divided"]
    assert not one.loc[3, "divided"] and not one.loc[4, "divided"]
    assert one.loc[2, "mother"] == 1 and one.loc[4, "mother"] == 2


def test_load_lineages(movie_dir):
    table = build_cell_table(str(movie_dir))
    lins = load_lineages("palbociclib", table)
    assert len(lins) == 1
    lin = lins[0]
    assert len(lin) == 5

    parents, daughters = lin.edges
    assert np.all(parents < daughters)  # mothers precede daughters
    assert sorted(zip(parents.tolist(), daughters.tolist(), strict=True)) == [(0, 1), (0, 2), (1, 3), (1, 4)]

    obs = lin.obs
    # Root: born before the drug and seen from birth, so it has a lifetime; it divided.
    assert obs[0, 1] == pytest.approx(70 * FRAME_HOURS) and obs[0, 2] == 1.0
    # Cell 1 divided after 60 frames; its sister was lost and so is censored.
    assert obs[1, 1] == pytest.approx(60 * FRAME_HOURS) and obs[1, 2] == 1.0
    assert obs[2, 2] == 0.0
    # Granddaughters: one tracked to the end, one lost, both censored.
    assert np.all(obs[3:, 2] == 0.0)
    assert obs[3, 1] == pytest.approx((N_FRAMES - DRUG_FRAME - 70) * FRAME_HOURS)


def test_cell_obs_unborn_and_artifacts():
    df = pd.DataFrame(
        {
            "mother": [0, 3, 3],
            "first": [0, 10, 10],
            "last": [9, 19, 19],
            "divided": [True, False, True],
            "cdk2_rate": [0.2, 0.3, 25.0],
            "s_entry_h": [np.nan, 1.0, np.nan],
        }
    )
    obs = cell_obs(df)
    # No birth seen: no lifetime or censoring flag.
    assert np.isnan(obs[0, 1]) and np.isnan(obs[0, 2])
    assert obs[1, 1] == pytest.approx(2.0) and obs[1, 2] == 0.0
    # A segmentation spike in the sensor ratio is dropped.
    assert np.isnan(obs[2, 0])


def test_shipped_table_loads():
    """The processed table in the package gives well-formed lineages for both conditions."""
    for cond in ("control", "palbociclib"):
        lins = load_lineages(cond)
        assert len(lins) > 100
        obs = np.vstack([lin.obs for lin in lins])
        born = np.isfinite(obs[:, 1])
        assert np.all(obs[born, 1] > 0)
        assert set(np.unique(obs[born, 2])) <= {0.0, 1.0}
        for lin in lins[:50]:
            parents, daughters = lin.edges
            assert np.all(parents < daughters)
            # Two daughters, or one when the sister never had a usable sensor frame.
            assert np.all(np.isin(np.bincount(parents, minlength=len(lin))[parents], (1, 2)))


def test_load_lineages_without_root_lifetimes(movie_dir):
    table = build_cell_table(str(movie_dir))
    [lin] = load_lineages("palbociclib", table, root_lifetimes=False)
    [full] = load_lineages("palbociclib", table)
    assert np.all(np.isnan(lin.obs[0, 1:3]))
    np.testing.assert_array_equal(lin.obs[0, 0], full.obs[0, 0])
    np.testing.assert_array_equal(lin.obs[1:], full.obs[1:])
