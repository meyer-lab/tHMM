"""Lineages under timed MEK or ERK inhibition, from Min et al. 2020 (EBI BioStudies S-BSST314).

Figure 1 of Min, Rong, Tian and Spencer, "Temporal integration of mitogen history in
mother cells controls proliferation of daughter cells", *Science* 368 (2020), is
deposited as

    https://ftp.ebi.ac.uk/biostudies/fire/S-BSST/S-BSSTxxx314/S-BSST314/Files/Min_Spencer_2020_Fig1.rar

(5.9 GB, RAR5; the movies are in its ``1CDEF_2BC_3D_S1DE_S2BDF`` folder). It is MCF10A
expressing the DHB CDK2 sensor, imaged every 12 min for 257 frames (the deposit's
``Data_organization.docx`` states the interval). Its ``plate_map.docx`` gives the design:
drug is added after frame 94 and washed out after 1, 3, 6 or 9 h, or left on.
Plate rows B-G (2-7) and columns 2-11 were filmed, four sites per well:

* columns 2-5: MEK inhibitor for 1, 3, 6, 9 h; columns 6-9: ERK inhibitor for 1, 3, 6, 9 h;
* column 10: MEK inhibitor left on (rows B-C) or ERK inhibitor left on (rows D-G);
* column 11: no drug.

That makes an exposure-duration series for each drug, with ~33 h of imaging after the
drug goes on, against ~23 h in the palbociclib movies of :mod:`.palbociclib_loader`.

The movie format is the one described in :mod:`.palbociclib_loader`, except that the DHB
nuclear and cytoplasmic-ring medians are in tracedata columns 11 and 13 (1-indexed).
:func:`build_cell_table` reduces the movies to a per-cell table, which ships with the
package (:data:`CELL_TABLE`).
"""

import glob
import os
import re

import numpy as np
import pandas as pd

from .LineageTree import LineageTree
from .palbociclib_loader import build_lineages, extract_movie
from .states.CensoredWeibullGaussian import StateDistribution

#: Index (0-based) of the first frame with drug present. The plate map says the drug went
#: on "at frame 94"; every well, control included, shows a one-frame handling dip in the
#: sensor ratio at 0-based frame 94, so that is the first frame imaged after addition.
DRUG_FRAME = 94
N_FRAMES = 257
NUC_COL, CYTO_COL = 10, 12

#: Hours of drug exposure for each condition; inf is drug left on, 0 no drug.
PULSES_H = (0.0, 1.0, 3.0, 6.0, 9.0, np.inf)
DRUGS = ("MEKi", "ERKi")

CELL_TABLE = os.path.join(os.path.dirname(__file__), "data", "MCF10A_mitogen", "fig1_cells.csv.gz")


def well_condition(row: int, col: int) -> tuple[str, float]:
    """(drug, exposure in hours) for a Figure 1 well; the control's drug is ``"none"``."""
    assert 2 <= row <= 7 and 2 <= col <= 11
    if col == 11:
        return "none", 0.0
    if col == 10:
        return ("MEKi" if row <= 3 else "ERKi"), np.inf
    return ("MEKi" if col <= 5 else "ERKi"), PULSES_H[1 + (col - 2) % 4]


def build_cell_table(fig1_dir: str) -> pd.DataFrame:
    """Reduce every Figure 1 movie to a per-cell table of cells born after the drug went
    on and their mothers, which is all :func:`load_lineages` uses.

    :param fig1_dir: the extracted ``Min_Spencer_2020_Fig1/1CDEF_2BC_3D_S1DE_S2BDF`` directory
    """
    tables = []
    for path in sorted(glob.glob(os.path.join(fig1_dir, "tracedata_*.mat"))):
        match = re.search(r"tracedata_(\d+)_(\d+)_(\d+)\.mat$", path)
        assert match is not None
        row, col, site = map(int, match.groups())
        df = extract_movie(path, N_FRAMES, NUC_COL, CYTO_COL)
        post = (df["mother"] > 0) & (df["first"] >= DRUG_FRAME)
        df = df[post | df["cell"].isin(df.loc[post, "mother"])]
        drug, pulse = well_condition(row, col)
        df.insert(0, "movie", f"{row}_{col}_{site}")
        df.insert(1, "drug", drug)
        df.insert(2, "pulse_h", pulse)
        tables.append(df)
    return pd.concat(tables, ignore_index=True)


def load_lineages(
    drug: str, pulse_h: float, table: pd.DataFrame | None = None, E=None, roots: str = "truncate"
) -> list[LineageTree]:
    """Lineages of cells born after drug addition in one condition.

    :param drug: ``"MEKi"`` or ``"ERKi"``; ignored when ``pulse_h`` is 0 (the no-drug wells)
    :param pulse_h: hours of exposure, one of :data:`PULSES_H`
    :param roots: how lineage roots, selected on dividing after the drug, are treated (see
        :func:`.palbociclib_loader.build_lineages`)
    """
    if table is None:
        table = pd.read_csv(CELL_TABLE)
    if E is None:
        E = [StateDistribution()]
    if pulse_h == 0.0:
        table = table[table["pulse_h"] == 0.0]
    else:
        table = table[(table["drug"] == drug) & (table["pulse_h"] == pulse_h)]
    assert len(table) > 0

    lineages = []
    for _, movie in table.groupby("movie", sort=True):
        lineages += build_lineages(movie, DRUG_FRAME, N_FRAMES, E, roots)
    return lineages


def generation_depths(lineages: list[LineageTree]) -> np.ndarray:
    """Generations below the root of every cell (the root is 0)."""
    out = []
    for lin in lineages:
        depth = np.zeros(len(lin), dtype=int)
        parents, daughters = lin.edges
        for p, d in zip(parents, daughters, strict=True):
            depth[d] = depth[p] + 1
        out.append(depth)
    return np.concatenate(out)
