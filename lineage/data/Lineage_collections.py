""" This file is to collect those lineages that have the same condition, and have 3 or greater number of cells in their lineages. """

from ..LineageInputOutput import import_Heiser
from ..states.StateDistributionGaPhs import StateDistribution
from ..LineageTree import LineageTree

#----------------------- Control conditions from both old and new versions -----------------------#
desired_num_states = 2
E = [StateDistribution() for i in range(desired_num_states)]


def popout_single_lineages(lineages):
    """ To remove lineages with cell numbers <= 7. """
    popped_lineages = []
    for ind, cells in enumerate(lineages):
        if len(cells) <= 7:
            pass
        else:
            popped_lineages.append(cells)
    return popped_lineages


# -- Lapatinib control

lap01 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00601_A5_1_V5.xlsx")]
lap02 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00701_A5_1_V4.xlsx")]
lap03 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00801_A5_1_V4.xlsx")]

lapControl = lap01 + lap02 + lap03
Lapatinib_Control = popout_single_lineages(lapControl)
# -- Gemcitabine control

gem04 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00601_A3_1_V4.xlsx")]
gem042 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00601_A3_2_V4.xlsx")]
gem05 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00701_A3_1_V5.xlsx")]
gem052 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00701_A3_2_V4.xlsx")]
gem06 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00801_A3_1_V4.xlsx")]
gem062 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00801_A3_2_V4.xlsx")]
gemControl = gem04 + gem05 + gem06 + gem042 + gem052 + gem062
Gemcitabine_Control = popout_single_lineages(gemControl)

#-------------------- GEMCITABINE 5 uMolars --------------------#

gemc33 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00601_C3_1_V4.xlsx")]
gemc34 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00701_C3_1_V4.xlsx")]
gemc35 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00801_C3_1_V5.xlsx")]
# replicates
gemc332 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00601_C3_2_V4.xlsx")]
gemc342 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00701_C3_2_V4.xlsx")]
gemc352 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00801_C3_2_V4.xlsx")]

gem5uM = gemc33 + gemc34 + gemc35 + gemc332 + gemc342 + gemc352

Gem5uM = popout_single_lineages(gem5uM)

#---------------------- LAPATINIB 25 uMolars ----------------------------#

lapb63 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00601_B6_1_V4.xlsx")]
lapb64 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00701_B6_1_V4.xlsx")]
lapb65 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00801_B6_1_V4.xlsx")]
# replicates
lapb632 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00601_B6_2_V4.xlsx")]
lapb642 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00701_B6_2_V4.xlsx")]
lapb652 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00801_B6_2_V4.xlsx")]

lap25uM = lapb63 + lapb64 + lapb65 + lapb632 + lapb642 + lapb652

Lapt25uM = popout_single_lineages(lap25uM)

#---------------- PACLITAXEL 2uMolars ---------------------#
# new
taxb40 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00602_B6_1_V4.xlsx")]
taxb402 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00602_B6_2_V4.xlsx")]
taxb41 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00702_B4_1_V4.xlsx")]
taxb412 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00702_B4_2_V4.xlsx")]
taxb42 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00802_B4_1_V4.xlsx")]
taxb422 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00802_B4_2_V4.xlsx")]

taxs = taxb40 + taxb402 + taxb41 + taxb412 + taxb42 + taxb422
Tax2uM = popout_single_lineages(taxs)
