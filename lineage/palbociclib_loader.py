"""Lineages, CDK2 activity, and censored lifetimes from Spencer-lab live-cell imaging under palbociclib.

Issue #1016 asked for MCF7/T47D palbociclib dose series from two Zenodo records
(10.5281/zenodo.10498701 and 10.5281/zenodo.12479168) and the GitHub repository
``sc-reporters/multiplex-imaging``. None of these holds such data: the first record is an
unrelated astrophysics thesis, the second is a deleted spam record, and the repository
does not exist. We are not aware of any public data set that combines lineage tracking,
a CDK2 sensor trace, and a palbociclib dose series in an ER+ breast cancer line.

The closest public data we found, and the one read here, is Figure 3A of Miller et al.,
"Ki67 is a graded rather than a binary marker of proliferation versus quiescence",
*Cell Reports* 24 (2018), deposited as EBI BioStudies S-BSST167:

    https://ftp.ebi.ac.uk/biostudies/fire/S-BSST/S-BSSTxxx167/S-BSST167/Files/u/Figure_3_Ki67.rar

(1.3 GB, RAR5). It is MCF10A (a non-transformed breast epithelial line) expressing the
DHB-mVenus CDK2 sensor, imaged for 248 frames. Drug is added at frame 135 (1-indexed),
and the plate rows are 1-2: MEK inhibitor 100 nM; 3-4: CDK4/6 inhibitor (palbociclib)
1 uM; 5: Nutlin-3 5 uM; 6-7: vehicle control. So there is one palbociclib dose, not a
series. The deposit does not state the frame interval; we use 12 min, the interval the
same lab reports for its identically formatted MCF10A movies (S-BSST314, S-BSST439), which
also gives the expected ~17 h median MCF10A cycle.

Each ``tracedata_{row}_{col}_{site}.mat`` file holds ``tracedata`` (cells x frames x
features, NaN where a cell is absent) and ``genealogy`` (each cell's mother id, 1-based,
NaN for none). A cell id spans one cycle: at mitosis the mother's track ends and two
daughters start on the next frame.

:func:`build_cell_table` reduces the raw movies to one row per cell, and a table for the
cells used in the analysis ships with the package (:data:`CELL_TABLE`), so the 1.3 GB
archive is only needed to rebuild it.
"""

import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.sparse import csr_array

from .LineageTree import LineageTree
from .states.CensoredWeibullGaussian import StateDistribution

FRAME_HOURS = 0.2
#: Index (0-based) of the first frame with drug present.
DRUG_FRAME = 134
N_FRAMES = 248
#: Figure 3A tracedata columns (0-based): DHB nuclear median, DHB cytoplasmic ring median.
NUC_COL, CYTO_COL = 5, 7

#: Plate row -> (condition, palbociclib dose in nM, or None for the other drugs).
FIG3A_ROWS = {
    1: ("MEKi", None),
    2: ("MEKi", None),
    3: ("palbociclib", 1000.0),
    4: ("palbociclib", 1000.0),
    5: ("Nutlin", None),
    6: ("control", 0.0),
    7: ("control", 0.0),
}

#: CDK2 activity (cytoplasm / nucleus) above which a cell is taken to have entered S
#: phase; CDK2-low quiescent cells sit near 0.4-0.5 in this data.
S_PHASE_CDK2 = 1.0

#: CDK2 activation rates beyond this (activity units per hour) come from single-frame
#: segmentation errors in the sensor ratio (the 99.5th percentile is ~1.8 in control),
#: so they are treated as missing.
MAX_RATE = 2.0

CELL_TABLE = os.path.join(os.path.dirname(__file__), "data", "MCF10A_palbociclib", "fig3a_cells.csv.gz")


def smooth(trace: np.ndarray, width: int = 5) -> np.ndarray:
    """Running median over ``width`` frames (1 h), ignoring gaps."""
    filled = pd.Series(trace).interpolate(limit_direction="both").to_numpy()
    return median_filter(filled, size=width, mode="nearest")


def max_activation_rate(cdk2: np.ndarray, skip: int = 5, window: int = 60, lag: int = 5, min_frames: int = 10) -> float:
    """Maximum rate of CDK2 activation during G1, in activity units per hour.

    The rate is the largest increase of the smoothed CDK2 activity over ``lag`` frames,
    taken from ``skip`` frames after birth (to step over the post-mitotic transient) until
    the cell reaches S-phase CDK2 levels, ``window`` frames pass, or the track ends.

    :param cdk2: one cell's CDK2 activity from birth, one entry per frame
    :return: the rate, or NaN when fewer than ``min_frames`` frames of G1 are available
    """
    s = smooth(cdk2)[skip:window]
    above = np.nonzero(s >= S_PHASE_CDK2)[0]
    if above.size:
        s = s[: above[0] + 1]
    if s.size < min_frames:
        return np.nan
    return float(np.max(s[lag:] - s[:-lag]) / (lag * FRAME_HOURS))


def s_phase_entry(cdk2: np.ndarray, skip: int = 5) -> float:
    """Hours from birth until the smoothed CDK2 activity first reaches S-phase levels, NaN if never."""
    s = smooth(cdk2)
    above = np.nonzero(s[skip:] >= S_PHASE_CDK2)[0]
    return float((above[0] + skip) * FRAME_HOURS) if above.size else np.nan


def extract_movie(
    path: str, n_frames: int = N_FRAMES, nuc_col: int = NUC_COL, cyto_col: int = CYTO_COL
) -> pd.DataFrame:
    """One row per cell of a single ``tracedata`` movie.

    :param nuc_col, cyto_col: tracedata columns (0-based) of the DHB nuclear and
        cytoplasmic-ring intensities, which differ between deposits
    """
    import scipy.io as sio

    m = sio.loadmat(path)
    trace, mother = m["tracedata"], m["genealogy"].ravel()
    assert trace.shape[1] == n_frames
    with np.errstate(divide="ignore", invalid="ignore"):
        cdk2 = trace[:, :, cyto_col] / trace[:, :, nuc_col]
    cdk2[~np.isfinite(cdk2)] = np.nan

    present = np.isfinite(cdk2)
    seen = present.any(axis=1)
    first = np.argmax(present, axis=1)
    last = n_frames - 1 - np.argmax(present[:, ::-1], axis=1)
    has_mother = np.isfinite(mother)
    divided = np.zeros(mother.size, dtype=bool)
    divided[mother[has_mother].astype(int) - 1] = True

    rows = []
    for i in np.nonzero(seen)[0]:
        tr = cdk2[i, first[i] : last[i] + 1]
        born = bool(has_mother[i])
        rows.append(
            {
                "cell": i + 1,
                "mother": int(mother[i]) if born else 0,
                "first": int(first[i]),
                "last": int(last[i]),
                "divided": bool(divided[i]),
                # G1 features only mean something when the track starts at birth.
                "cdk2_rate": max_activation_rate(tr) if born else np.nan,
                "s_entry_h": s_phase_entry(tr) if born else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_cell_table(fig3a_dir: str, post_drug_only: bool = True) -> pd.DataFrame:
    """Reduce every Figure 3A movie to a per-cell table.

    :param fig3a_dir: the extracted ``Figure_3/Figure_3A`` directory
    :param post_drug_only: keep only cells born after drug addition and their mothers,
        which is all :func:`load_lineages` uses
    """
    tables = []
    for path in sorted(glob.glob(os.path.join(fig3a_dir, "tracedata_*.mat"))):
        match = re.search(r"tracedata_(\d+)_(\d+)_(\d+)\.mat$", path)
        assert match is not None
        row, col, site = map(int, match.groups())
        df = extract_movie(path)
        if post_drug_only:
            post = (df["mother"] > 0) & (df["first"] >= DRUG_FRAME)
            keep = post | df["cell"].isin(df.loc[post, "mother"])
            df = df[keep]
        condition, dose = FIG3A_ROWS[row]
        df.insert(0, "movie", f"{row}_{col}_{site}")
        df.insert(1, "condition", condition)
        df.insert(2, "dose_nM", dose)
        tables.append(df)
    return pd.concat(tables, ignore_index=True)


#: Observation column holding each cell's S-phase entry time (see :func:`cell_obs`).
S_ENTRY_COL = 5


def cell_obs(df: pd.DataFrame) -> np.ndarray:
    """Observation rows ``[cdk2_rate, lifetime_h, divided, t_lo, t_hi, s_entry_h]`` for cells in ``df``.

    A cell whose birth was not seen has no defined lifetime or G1 feature, so both are
    NaN, as is an implausible activation rate (see :data:`MAX_RATE`). Otherwise the lifetime runs from birth to division, or to the last frame the cell
    was tracked, where it is right-censored. That treats cells lost from tracking the same
    as cells still undivided at the end of the movie, which assumes the loss is unrelated
    to when the cell would have divided. ``t_lo`` and ``t_hi`` are the lifetime truncation
    window, NaN here and set for lineage roots by :func:`build_lineages`. The last column
    is ignored by the emission and only used to define escape for
    :mod:`lineage.early_biomarker`.
    """
    born = (df["mother"] > 0).to_numpy()
    life = ((df["last"] - df["first"] + 1) * FRAME_HOURS).to_numpy(dtype=float)
    rate = df["cdk2_rate"].to_numpy(dtype=float)
    return np.column_stack(
        [
            np.where(np.abs(rate) <= MAX_RATE, rate, np.nan),
            np.where(born, life, np.nan),
            np.where(born, df["divided"].to_numpy(dtype=float), np.nan),
            np.full((len(df), 2), np.nan),
            df["s_entry_h"].to_numpy(dtype=float),
        ]
    )


#: How :func:`build_lineages` treats the lifetimes of lineage roots.
ROOT_MODES = ("truncate", "keep", "drop")


def build_lineages(
    movie: pd.DataFrame, drug_frame: int, n_frames: int, E, roots: str = "truncate"
) -> list[LineageTree]:
    """Lineages of cells born at or after ``drug_frame`` in one movie's cell table.

    Each lineage is rooted at a cell that divided at or after ``drug_frame``; its daughters,
    and any of their descendants, were born into drug. The root keeps its own observation
    when its birth was seen (its G1 was before the drug), so the root-to-daughter
    transitions describe how a cell's pre-treatment state carries into its daughters'
    response. A mother with a single daughter here (the sister never had a usable sensor
    frame) keeps just the one, rather than an imputed sister.

    :param roots: a root is only in the data because it divided between ``drug_frame`` and
        the end of the movie, so its lifetime is selected on the outcome. ``"truncate"``
        (the default) conditions the root's lifetime on that window, which is the exact
        likelihood given selection; ``"keep"`` ignores the selection; ``"drop"`` removes the
        roots' lifetimes (keeping their biosensor readings).
    """
    assert roots in ROOT_MODES
    movie = movie.set_index("cell")
    post = movie[(movie["mother"] > 0) & (movie["first"] >= drug_frame)]
    children: dict[int, list[int]] = {}
    for cell, mom in zip(post.index, post["mother"], strict=True):
        children.setdefault(int(mom), []).append(int(cell))

    lineages = []
    for root in sorted(m for m in children if m not in post.index):
        # Breadth-first, so that every mother precedes her daughters.
        order, parent_pos = [root], [-1]
        k = 0
        while k < len(order):
            for c in sorted(children.get(order[k], [])):
                order.append(c)
                parent_pos.append(k)
            k += 1

        n = len(order)
        tree = csr_array((np.ones(n - 1, dtype=bool), (np.array(parent_pos[1:]), np.arange(1, n))), shape=(n, n))
        obs = cell_obs(movie.reindex(order).reset_index())
        if root not in movie.index:
            # A mother with no usable sensor frames at all is kept for her topology only.
            obs[0, :] = np.nan
        elif roots == "drop":
            obs[0, 1:3] = np.nan
        elif roots == "truncate" and np.isfinite(obs[0, 1]):
            # Daughters born in [drug_frame, n_frames - 1] means a lifetime, in frames,
            # between these bounds.
            first = movie.loc[root, "first"]
            obs[0, 3:5] = np.array([drug_frame - first, n_frames - 1 - first]) * FRAME_HOURS
        lineages.append(LineageTree(tree, E, obs=obs))
    return lineages


def load_lineages(
    condition: str, table: pd.DataFrame | None = None, E=None, roots: str = "truncate"
) -> list[LineageTree]:
    """Lineages of cells born after drug addition in one condition (see :func:`build_lineages`).

    :param condition: ``"control"``, ``"palbociclib"``, ``"MEKi"``, or ``"Nutlin"``
    :param table: a table from :func:`build_cell_table`; defaults to :data:`CELL_TABLE`
    :param E: emissions to attach to the lineages (only used as a template for fitting)
    """
    if table is None:
        table = pd.read_csv(CELL_TABLE)
    if E is None:
        E = [StateDistribution()]
    table = table[table["condition"] == condition]
    lineages = []
    for _, movie in table.groupby("movie", sort=True):
        lineages += build_lineages(movie, DRUG_FRAME, N_FRAMES, E, roots)
    return lineages
