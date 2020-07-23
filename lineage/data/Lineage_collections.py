""" This file is to collect those lineages that have the same condition, and have 3 or greater number of cells in their lineages. """

from ..LineageInputOutput import import_Heiser
from ..states.StateDistributionGaPhs import StateDistribution
from ..LineageTree import LineageTree

#----------------------- Control conditions from both old and new versions -----------------------#
desired_num_states = 2
E = [StateDistribution() for i in range(desired_num_states)]


def popout_single_lineages(lineages):
    """ To remove lineages with cell numbers <= 3. """
    popped_lineages = []
    for ind, cells in enumerate(lineages):
        if len(cells) <= 7:
            pass
        else:
            popped_lineages.append(cells)
    return popped_lineages


# -- Lapatinib control
lap01 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/old_version/AU00601_A5_1.xlsx")]
lap02 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/old_version/AU00701_A5_1.xlsx")]
lap03 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/old_version/AU00801_A5_1_.xlsx")]
lap04 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00601_A5_1_V4.xlsx")]
lap05 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00701_A5_1_V4.xlsx")]
lap06 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00801_A5_1_V4.xlsx")]

lapControl = lap01 + lap02 + lap03 + lap04 + lap05 + lap06
Lapatinib_Control = popout_single_lineages(lapControl)
# -- Gemcitabine control
# old
gem01 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/old_version/AU00601_A3_1.xlsx")]
gem012 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/old_version/AU00601_A3_2.xlsx")]
gem02 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/old_version/AU00701_A3_1.xlsx")]
gem022 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/old_version/AU00701_A3_2.xlsx")]
gem03 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/old_version/AU00801_A3_1.xlsx")]
gem032 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/old_version/AU00801_A3_2.xlsx")]
# new
gem04 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00601_A3_1_V4.xlsx")]
gem042 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00601_A3_2_V4.xlsx")]
gem05 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00701_A3_1_V4.xlsx")]
gem052 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00701_A3_2_V4.xlsx")]
gem06 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00801_A3_1_V4.xlsx")]
gem062 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00801_A3_2_V4.xlsx")]
gemControl = gem01 + gem02 + gem03 + gem04 + gem05 + gem06 + gem012 + gem022 + gem032 + gem042 + gem052 + gem062
Gemcitabine_Control = popout_single_lineages(gemControl)

#-------------------- GEMCITABINE 5 uMolars --------------------#
# old
gemc30 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/old_version/AU00601_C3_1.xlsx")]
gemc31 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/old_version/AU00701_C3_1.xlsx")]
gemc32 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/old_version/AU00801_C3_1_.xlsx")]
# replicates
gemc302 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/old_version/AU00601_C3_2.xlsx")]
gemc312 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/old_version/AU00701_C3_2.xlsx")]
gemc322 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/old_version/AU00801_C3_2_.xlsx")]
# new
gemc33 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00601_C3_1_V4.xlsx")]
gemc34 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00701_C3_1_V4.xlsx")]
gemc35 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00801_C3_1_V4.xlsx")]
# replicates
gemc332 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00601_C3_2_V4.xlsx")]
gemc342 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00701_C3_2_V4.xlsx")]
gemc352 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00801_C3_2_V4.xlsx")]

gem5uM = gemc30 + gemc31 + gemc32 + gemc302 + gemc312 + gemc322 + gemc33 + gemc34 + gemc35 + gemc332 + gemc342 + gemc352

Gem5uM = popout_single_lineages(gem5uM)

#---------------------- LAPATINIB 25 uMolars ----------------------------#
# old
lapb60 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/old_version/AU00601_B6_1.xlsx")]
lapb61 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/old_version/AU00701_B6_1.xlsx")]
lapb62 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/old_version/AU00801_B6_1.xlsx")]
# replicates
lapb602 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/old_version/AU00601_B6_2.xlsx")]
lapb612 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/old_version/AU00701_B6_2.xlsx")]
lapb622 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/old_version/AU00801_B6_2.xlsx")]
# new
lapb63 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00601_B6_1_V4.xlsx")]
lapb64 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00701_B6_1_V4.xlsx")]
lapb65 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00801_B6_1_V4.xlsx")]
# replicates
lapb632 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00601_B6_2_V4.xlsx")]
lapb642 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00701_B6_2_V4.xlsx")]
lapb652 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00801_B6_2_V4.xlsx")]

lap25uM = lapb60 + lapb61 + lapb62 + lapb602 + lapb612 + lapb622 + lapb63 + lapb64 + lapb65 + lapb632 + lapb642 + lapb652

Lapt25uM = popout_single_lineages(lap25uM)

#---------------- PACLITAXEL 2uMolars ---------------------#
# new
taxb40 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00602_B6_1_V4.xlsx")]
taxb402 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00602_B6_2_V4.xlsx")]
taxb41 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00702_B4_1_V4.xlsx")]
taxb412 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00702_B4_2_V4.xlsx")]
taxb42 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00802_B4_1_V4.xlsx")]
taxb422 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/new_version/AU00802_B4_2_V4.xlsx")]
# old
taxb45 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/old_version/AU00602_B6_1.xlsx")]
taxb452 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/old_version/AU00602_B6_2.xlsx")]
taxb46 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/old_version/AU00702_B4_1.xlsx")]
taxb462 = [LineageTree(list_of_cells, E) for list_of_cells in import_Heiser(path=r"lineage/data/heiser_data/old_version/AU00702_B4_2.xlsx")]

taxs = taxb40 + taxb402 + taxb41 + taxb412 + taxb42 + taxb422 + taxb45 + taxb452 + taxb46 + taxb462
Tax2uM = popout_single_lineages(taxs)
