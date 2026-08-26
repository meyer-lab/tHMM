"""This file is to collect those lineages that have the same condition, and have 3 or greater number of cells in their lineages."""

from .import_lineage import MCF10A, import_taxol
from .LineageInputOutput import import_exp_data
from .LineageTree import LineageTree
from .states.StateDistributionGamma import StateDistribution as StDist
from .states.StateDistributionGaPhs import StateDistribution

# ----------------------- Control conditions from both old and new versions -----------------------#
desired_num_states = 2
E = [StateDistribution() for _ in range(desired_num_states)]
E2 = [StDist() for _ in range(desired_num_states)]


def popout_single_lineages(lineages):
    """To remove lineages with cell numbers <= 5."""
    trimed_lineages = []
    for cells in lineages:
        if len(cells) < 5:
            pass
        else:
            trimed_lineages.append(cells)
    assert len(trimed_lineages) > 0
    return trimed_lineages


def load_condition(filename: str) -> list[LineageTree]:
    return [
        LineageTree(list_of_cells, E)
        for list_of_cells in import_exp_data(path=f"lineage/data/LineageData/{filename}.xlsx")
    ]


def load_conditions(filenames: list[str]) -> list[LineageTree]:
    trees = []
    for f in filenames:
        trees.extend(load_condition(f))
    return trees


def load_replicate_groups(replicate_file_groups: list[list[str]]):
    all_trees_by_file: list[list[LineageTree]] = []
    reps: list[int] = []
    flat_all_trees: list[LineageTree] = []
    for group in replicate_file_groups:
        group_trees: list[LineageTree] = []
        for filename in group:
            trees = load_condition(filename)
            all_trees_by_file.append(trees)
            group_trees.extend(trees)
        reps.append(len(group_trees))
        flat_all_trees.extend(group_trees)
    lengths_by_file = [len(t) for t in all_trees_by_file]
    return flat_all_trees, lengths_by_file, reps


# -- Lapatinib conditions
Lapatinib_Control, len_lp_cntr, lpt_cn_reps = load_replicate_groups(
    [
        ["AU00601_A5_1_V5", "AU00601_A5_2_V4"],
        ["AU00701_A5_1_V4"],
        ["AU00801_A5_1_V4"],
    ]
)

Lapt25uM, len_lp_25, lpt_25_reps = load_replicate_groups(
    [
        ["AU00601_B6_1_V4", "AU00601_B6_2_V4"],
        ["AU00701_B6_1_V4", "AU00701_B6_2_V4"],
        ["AU00801_B6_1_V4", "AU00801_B6_2_V4", "AU00801_B6_3_V4"],
    ]
)

Lapt50uM, len_lp_50, lpt_50_reps = load_replicate_groups(
    [
        [
            "AU00601_C5_1_V4",
            "AU00601_C5_2_1_V4",
            "AU00601_C5_2_2_V4",
            "AU00601_C5_3_1",
            "AU00601_C5_3_2",
            "AU00601_C5_3_3",
        ],
        ["AU00701_C5_1_V4", "AU00701_C5_2_V4", "AU00701_C5_3_V4", "AU00701_C5_4_V4"],
        ["AU00801_C5_1_V4", "AU00801_C5_2_V4", "AU00801_C5_3_1", "AU00801_C5_3_2"],
    ]
)

Lap250uM, len_lp_250, lpt_250_reps = load_replicate_groups(
    [
        ["AU00601_D5_1_V4", "AU00601_D5_2_V4", "AU00601_D5_3_V4"],
        ["AU00701_D5_1_V4", "AU00701_D5_2_V4"],
        ["AU00801_D5_1_V4", "AU00801_D5_2_V4", "AU00801_D5_3_V4"],
    ]
)

# -- Gemcitabine conditions
Gemcitabine_Control, len_gm_cntr, gem_cn_reps = load_replicate_groups(
    [
        ["AU00601_A3_1_V4", "AU00701_A3_1_V5", "AU00801_A3_1_V4"],
        ["AU00601_A3_2_V4", "AU00701_A3_2_V4"],
        ["AU00801_A3_2_V4", "AU00601_A3_3_V4"],
    ]
)

Gem5uM, len_gm_5, gem_5_reps = load_replicate_groups(
    [
        ["AU00601_C3_1_V4", "AU00601_C3_2_V4"],
        ["AU00701_C3_1_V4", "AU00701_C3_2_V4"],
        ["AU00801_C3_1_V5", "AU00801_C3_2_V4"],
    ]
)
gem5uM = popout_single_lineages(Gem5uM)

Gem10uM, len_gm_10, gem_10_reps = load_replicate_groups(
    [
        [
            "AU00601_C4_1_V5",
            "AU00601_C4_2_1_V4",
            "AU00601_C4_2_2_V4",
            "AU00601_C4_3_1_V5",
            "AU00601_C4_3_2_V4",
            "AU00601_C4_4_V4",
        ],
        ["AU00701_C4_1_V4", "AU00701_C4_2_V4", "AU00701_C4_3_V4", "AU00701_C4_4_V4"],
        ["AU00801_C4_1_V4", "AU00801_C4_2_V4", "AU00801_C4_3_V4", "AU00801_C4_4_1", "AU00801_C4_4_2"],
    ]
)

Gem30uM, len_gm_30, gem_30_reps = load_replicate_groups(
    [
        ["AU00601_D3_1_V4", "AU00601_D3_2_V4", "AU00601_D3_3_V4"],
        ["AU00701_D3_1_V4", "AU00701_D3_2_V4"],
        ["AU00801_D3_1_V4", "AU00801_D3_2_V4", "AU00801_D3_3_V4"],
    ]
)

# -- Other drug conditions
Tax2uM = load_conditions(
    ["AU00602_B6_1_V4", "AU00602_B6_2_V4", "AU00702_B4_1_V4", "AU00702_B4_2_V4", "AU00802_B4_1_V4", "AU00802_B4_2_V4"]
)
Tax7uM = load_conditions(["AU00602_D3_1_V4", "AU00602_D3_2_V4"])
Palbo250uM = load_conditions(
    ["AU00602_D3_1_V4", "AU00702_D1_1_V4", "AU00802_D1_1_V4", "AU00602_D3_2_V4", "AU00702_D1_2_V4", "AU00802_D1_2_V4"]
)

# -- MCF10A conditions
pbs = [LineageTree(cells, E2) for cells in MCF10A("PBS")]
egf = [LineageTree(cells, E2) for cells in MCF10A("EGF")]
hgf = [LineageTree(cells, E2) for cells in MCF10A("HGF")]
osm = [LineageTree(cells, E2) for cells in MCF10A("OSM")]

AllLapatinib = [Lapatinib_Control + Gemcitabine_Control, Lapt25uM, Lapt50uM, Lap250uM]
AllGemcitabine = [Lapatinib_Control + Gemcitabine_Control, Gem5uM, Gem10uM, Gem30uM]
GFs = [pbs, egf, hgf, osm]

##########################################################
# NEW PACLITAXEL DATA
untreated, taxol_05, taxol_1, taxol_15, taxol_2, taxol_25, taxol_3, taxol_4 = import_taxol()
# 555, 1119, 486, 382, 272, 365, 246, 266
untreated_taxol = [LineageTree(list_cells, E) for list_cells in untreated] # 555 cells
Taxol_1 = [LineageTree(list_cells, E) for list_cells in taxol_1] # 486 cells
Taxol_2 = [LineageTree(list_cells, E) for list_cells in taxol_2] # 272 cells
Taxol_3 = [LineageTree(list_cells, E) for list_cells in taxol_3] # 246 cells
Taxol_4 = [LineageTree(list_cells, E) for list_cells in taxol_4] # 266 cells

taxols = [untreated_taxol, Taxol_1, Taxol_2, Taxol_4]