""" This file includes functions to import the new lineage data. """
import pandas as pd
import itertools as it
import numpy as np
from .CellVar import CellVar

############################
# importing AU565 data (new)
############################
# path = "lineage/data/LineageData/AU02101_A3_field_1_RP_50_CSV-Table.csv"


def import_AU565(path: str) -> list:
    """ Importing AU565 file cells.
    :param path: the path to the file.
    :return population: list of cells structured in CellVar objects.
    """
    df = pd.read_csv(path)

    population = []
    # loop over "lineageId"s
    for i in df["lineageId"].unique():
        # select all the cells that belong to that lineage
        lineage = df.loc[df['lineageId'] == i]

        # if the lineage Id exists, do the rest, if not, pass
        if not(lineage.empty):
            unique_cell_ids = list(lineage["trackId"].unique())  # the length of this shows the number of cells in this lineage
            unique_parent_trackIDs = lineage["parentTrackId"].unique()

            pid = [[0]]  # root parent's parent id
            for j in unique_parent_trackIDs:
                if j != 0:
                    pid.append(np.count_nonzero(lineage["parentTrackId"] == j) * [j])
            parent_ids = list(it.chain(*pid))

            # create the root parent cell and assign obsrvations
            parent_cell = CellVar(parent=None)
            parent_cell = assign_observs_AU565(parent_cell, lineage, unique_cell_ids[0])

            # create a list to store cells belonging to a lineage
            lineage_list: list[CellVar] = [parent_cell]
            for k, val in enumerate(unique_cell_ids):
                if val in parent_ids:  # if the id of a cell exists in the parent ids, it means the cell divides
                    parent_index = [indx for indx, value in enumerate(parent_ids) if value == val]  # find whose mother it is
                    assert len(parent_index) == 2  # make sure has two children
                    a = assign_observs_AU565(CellVar(parent=lineage_list[k]), lineage, unique_cell_ids[parent_index[0]])
                    b = assign_observs_AU565(CellVar(parent=lineage_list[k]), lineage, unique_cell_ids[parent_index[1]])
                    lineage_list[k].left = a
                    lineage_list[k].right = b

                    lineage_list.append(a)
                    lineage_list.append(b)

        assert len(lineage_list) == len(unique_cell_ids)
        # if both observations are zero, remove the cell
        for n, cell in enumerate(lineage_list):
            if (cell.obs[1] == 0 and cell.obs[2] == 0):
                lineage_list.pop(n)

        population.append(lineage_list)
    return population


def assign_observs_AU565(cell: CellVar, lineage, uniq_id: int) -> CellVar:
    """Given a cell, the lineage, and the unique id of the cell, it assigns the observations of that cell, and returns it.
    :param cell: a CellVar object to be assigned observations.
    :param lineage: the lineage list of cells that the given cell is from.
    :param uniq_id: the id given to the cell from the experiment.
    """
    # initialize
    cell.obs = np.array([1, 0, 0, 0], dtype=float)
    parent_id = lineage["parentTrackId"].unique()
    # cell fate: die = 0, divide = 1
    if not(uniq_id in parent_id):  # if the cell has not divided, means either died or reached experiment end time
        if np.max(lineage.loc[lineage["trackId"] == uniq_id]["frame"]) == 49:  # means reached end of experiment
            cell.obs[0] = np.nan  # don't know
            cell.obs[3] = 1  # censored
        else:  # means cell died before experiment ended
            cell.obs[0] = 0  # died
            cell.obs[3] = 0  # not censored

    if cell.gen == 1:  # it is root parent
        cell.obs[3] = 1  # meaning it is left censored in its lifetime

    # cell's lifetime
    cell.obs[1] = 0.5 * (np.max(lineage.loc[lineage['trackId'] == uniq_id]['frame']) - np.min(lineage.loc[lineage['trackId'] == uniq_id]['frame']))
    # cell's diameter
    diam = np.array(lineage.loc[lineage['trackId'] == uniq_id]['Diameter_0'])
    if np.count_nonzero(diam == 0.0) != 0:
        diam[diam == 0.0] = np.nan
        if diam.size == 0:  # if empty
            cell.obs[2] = np.nan
        elif np.sum(np.isfinite(diam)) == 0:  # if all are nan
            cell.obs[2] = np.nan
        else:
            cell.obs[2] = np.nanmean(diam)
    else:
        cell.obs[2] = np.mean(diam)

    return cell

#######################
# importing MCF10A data
#######################
# partof_path = "lineage/data/MCF10A/"


def import_MCF10A(path: str):
    """ Reading the data and extracting lineages and assigning their corresponding observations.
    :param path: the path to the mcf10a data.
    :return population: list of cells structured in CellVar objects.
    """
    df = pd.read_csv(path)
    population = []
    # loop over "lineageId"s
    for i in df["lineage"].unique():
        # select all the cells that belong to that lineage
        lineage = df.loc[df['lineage'] == i]

        lin_code = list(lineage["TID"].unique())[0]  # lineage code to process
        unique_parent_trackIDs = lineage["motherID"].unique()

        parent_cell = CellVar(parent=None)
        parent_cell = assign_observs_MCF10A(parent_cell, lineage, lin_code)

        # create a list to store cells belonging to a lineage
        lineage_list = [parent_cell]
        for k, val in enumerate(unique_parent_trackIDs[1:]):
            temp_lin = lineage.loc[lineage["motherID"] == val]
            child_id = temp_lin["TID"].unique()  # find children
            if not (len(child_id) == 2):
                break
                lineage_list = []
            for cells in lineage_list:
                if lin_code == val:
                    cell = cells

            a = assign_observs_MCF10A(CellVar(parent=cell), lineage, child_id[0])
            b = assign_observs_MCF10A(CellVar(parent=cell), lineage, child_id[1])
            cell.left = a
            cell.right = b

            lineage_list.append(a)
            lineage_list.append(b)

        if lineage_list:
            # organize the order of cells by their generation
            ordered_list = []
            max_gen = np.max(lineage["generation"])
            for ii in range(1, max_gen + 1):
                for cells in lineage_list:
                    if cells.gen == ii:
                        ordered_list.append(cells)

            population.append(ordered_list)
    return population


def assign_observs_MCF10A(cell, lineage, uniq_id: int):
    """Given a cell, the lineage, and the unique id of the cell, it assigns the observations of that cell, and returns it.
    :param cell: a CellVar object to be assigned observations.
    :param lineage: the lineage list of cells that the given cell is from.
    :param uniq_id: the id given to the cell from the experiment.
    """
    # initialize
    cell.obs = [1, 0, 1, 0, 0]  # [fate, lifetime, censored?, velocity, mean_distance]
    t_end = 2880
    # check if cell's lifetime is zero
    if (np.max(lineage.loc[lineage['TID'] == uniq_id]['tmin']) - np.min(lineage.loc[lineage['TID'] == uniq_id]['tmin'])) / 60 == 0:
        lineage = lineage.loc[lineage["tmin"] < 2880]
        t_end = 2850
    parent_id = lineage["motherID"].unique()

    # cell fate: die = 0, divide = 1
    if not(uniq_id in parent_id):  # if the cell has not divided, means either died or reached experiment end time
        if np.max(lineage.loc[lineage["TID"] == uniq_id]["tmin"]) == t_end:  # means reached end of experiment
            cell.obs[0] = np.nan  # don't know
            cell.obs[2] = 0  # censored
        else:  # means cell died before experiment ended
            cell.obs[0] = 0  # died
            cell.obs[2] = 1  # not censored

    if cell.gen == 1:  # it is root parent
        cell.obs[2] = 0  # meaning it is left censored in its lifetime

    # cell's lifetime
    cell.obs[1] = (np.max(lineage.loc[lineage['TID'] == uniq_id]['tmin']) - np.min(lineage.loc[lineage['TID'] == uniq_id]['tmin'])) / 60
    cell.obs[3] = np.mean(lineage.loc[lineage['TID'] == uniq_id]['average_velocity'])
    cell.obs[4] = np.mean(lineage.loc[lineage['TID'] == uniq_id]['distance_mean'])

    return cell


def MCF10A(condition: str):
    """ Creates the population of lineages for each condition.
    Conditions include: PBS, EGF-treated, HGF-treated, OSM-treated.
    :param condition: a condition between [PBS, EGF, HGF, OSM]
    """
    if condition == "PBS":
        data1 = import_MCF10A("lineage/data/MCF10A/PBS_1.csv")
        data2 = import_MCF10A("lineage/data/MCF10A/PBS_2.csv")
        return data1 + data2

    elif condition == "EGF":
        data1 = import_MCF10A("lineage/data/MCF10A/EGF_1.csv")
        data2 = import_MCF10A("lineage/data/MCF10A/EGF_2.csv")
        data3 = import_MCF10A("lineage/data/MCF10A/EGF_3.csv")
        return data1 + data2 + data3

    elif condition == "HGF":
        data1 = import_MCF10A("lineage/data/MCF10A/HGF_1.csv")
        data2 = import_MCF10A("lineage/data/MCF10A/HGF_2.csv")
        data3 = import_MCF10A("lineage/data/MCF10A/HGF_3.csv")
        data4 = import_MCF10A("lineage/data/MCF10A/HGF_4.csv")
        data5 = import_MCF10A("lineage/data/MCF10A/HGF_5.csv")
        return data1 + data2 + data3 + data4 + data5

    elif condition == "OSM":
        data1 = import_MCF10A("lineage/data/MCF10A/OSM_1.csv")
        data2 = import_MCF10A("lineage/data/MCF10A/OSM_2.csv")
        data3 = import_MCF10A("lineage/data/MCF10A/OSM_3.csv")
        data4 = import_MCF10A("lineage/data/MCF10A/OSM_4.csv")
        data5 = import_MCF10A("lineage/data/MCF10A/OSM_5.csv")
        data6 = import_MCF10A("lineage/data/MCF10A/OSM_6.csv")
        return data1 + data2 + data3 + data4 + data5 + data6

    else:
        raise ValueError("condition does not exist. choose between [PBS, EGF, HGF, OSM]")


def assign_observs_Taxol(cell, cell_lineage, label: int):
    """Given a cell, the lineage, and the unique id of the cell, it assigns the observations of that cell, and returns it.
    :param cell: a CellVar object to be assigned observations.
    :param cell_lineage: the dataframe of cells in a file.
    :param label: the id given to the cell from the experiment.
    """
    # initialize
    cell.obs = [1, 1, 0, 0, 1, 1]  # [G1_fate, S_G2_fate, G1_length, SG2_length, G1_censored?, SG2_censored?]
    t_end = 5760

    all_parent_ids = cell_lineage["parent"].unique()
    cell_df = cell_lineage.loc[cell_lineage["label"] == label]
    parent_id = cell_df["parent"].unique()[0]
    if parent_id == 0:
        cell.gen = 1

    threshold = 0.7 # greater than this ==> SG2
    G1_df = cell_df.loc[cell_df['Cell_CC_mean_intensity_ratio'] <= threshold]
    SG2_df = cell_df.loc[cell_df['Cell_CC_mean_intensity_ratio'] > threshold]

    ## G1 calculations:
    if G1_df.empty:
        # means the cell has no G1 component
        cell.obs[0] = 1.0 # if no G1, then there is SG2, so it has definitely transitioned from G1 to SG2
        cell.obs[2] = np.nan
        cell.obs[4] = np.nan
    else:
        # G1 length
        cell.obs[2] = (np.nanmax(G1_df["elapsed_minutes"]) - np.nanmin(G1_df["elapsed_minutes"])) / 60 # G1 duration
        if np.nanmin(G1_df["elapsed_minutes"]) == 0: # means left censored
            cell.obs[4] = 0
            if not(SG2_df.empty): # has transitioned from G1 -> SG2
                cell.obs[0] = 1

        if np.nanmax(G1_df["elapsed_minutes"]) == t_end: # cell existed and the experiment ended
            cell.obs[4] = 0
            assert not(label in all_parent_ids) # definitely is not a parent because we reached end of exp
            cell.obs[0] = np.nan

        if np.nanmax(G1_df["elapsed_minutes"]) < t_end and SG2_df.empty: # cell died in G1
            cell.obs[0] = 0

    ## SG2 calculations:
    if SG2_df.empty:
        # means the cell has no SG2 component
        cell.obs[1] = np.nan
        cell.obs[3] = np.nan
        cell.obs[5] = np.nan
    else:
        # SG2 length
        cell.obs[3] = (np.nanmax(SG2_df["elapsed_minutes"]) - np.nanmin(SG2_df["elapsed_minutes"])) / 60 # SG2
        if np.nanmin(SG2_df["elapsed_minutes"]) == 0: # cell existed when the experiment started
            cell.obs[5] = 0
            if label in all_parent_ids:
                cell.obs[1] = 1

        if np.nanmax(SG2_df["elapsed_minutes"]) == t_end: # cell existed and the experiment ended
            cell.obs[5] = 0
            assert not(label in all_parent_ids) # definitely is not a parent because we reached end of exp
            cell.obs[1] = np.nan

        if np.nanmax(SG2_df["elapsed_minutes"]) < t_end and not(label in all_parent_ids): # cell died in SG2
            cell.obs[1] = 0

    return cell


### The following two functions are used to make a list from labels that belong to the same lineage
def sep_lineages(data):
    labels = list(data["label"].unique())
    all_parent_ids = list(data["parent"].unique())
    a = []
    for i, val in enumerate(labels):
        df = data.loc[data["label"]==val] # cell itself
        if np.all(df["is_parent"] == True):
            df_parent = data.loc[data["parent"] == val] # cell's parent
            assert val in all_parent_ids
            if len(df_parent["label"].unique()) == 2:
                a.append([[val]] + [list(df_parent["label"].unique())])
        else:
            a.append([[val]])

    out1 = separate_lineages(data, all_parent_ids, a, 2)
    out2 = separate_lineages(data, all_parent_ids, out1, 3)
    out3 = separate_lineages(data, all_parent_ids, out2, 4)
    out4 = separate_lineages(data, all_parent_ids, out3, 5)
    out5 = separate_lineages(data, all_parent_ids, out4, 6)

    return out5

def separate_lineages(data, all_parent_ids, ls, k):
    lss = ls.copy()
    for j, val in enumerate(ls):
        if len(val) == k:
            temp = []
            k_th = val[k-1]
            for i in k_th:
                if np.isnan(i):
                    temp += [np.nan, np.nan]
                else:
                    if i in all_parent_ids:
                        if len(data.loc[data["parent"]==i]["label"].unique()) == 2:
                            temp += list(data.loc[data["parent"]==i]["label"].unique())
                        else:
                            temp += [np.nan, np.nan]
                    else:
                        temp += [np.nan, np.nan]
        else:
            continue
        if np.all(np.isnan(temp)):
            pass
        else:
            lss[j].append(temp)
    return lss


def import_taxol_file(filename="HC00801_A1_field_1_level_1.csv"):
    """To import the new Taxol data"""

    data = pd.read_csv("lineage/data/taxol/"+filename)

    lineage_codes = sep_lineages(data)
    lineages = []
    for i, lin in enumerate(lineage_codes):
        num_gens = len(lin) # number of generations
        parent_cell = CellVar(parent=None)
        parent_cell = assign_observs_Taxol(parent_cell, data, lin[0][0])
        lin_temp = [[parent_cell]]

        if num_gens >= 2:
            for kk in range(1, num_gens):
                tmp = []
                for ix, l in enumerate(lin[kk]):
                    if ix % 2 == 1: # only iterate through one of two daughter cells
                        pass
                    else:
                        if not np.isnan(l):
                            cell = lin_temp[kk-1][int(ix/2)]
                            cell.left = assign_observs_Taxol(cell, data, l)
                            cell.right = assign_observs_Taxol(cell, data, lin[kk][ix+1])
                            tmp.append([cell.left, cell.right])
                        else:
                            tmp.append([np.nan, np.nan])
                lin_temp.append(list(it.chain(*tmp)))
        lineages.append(lin_temp)
    return lineages

def import_taxol():
    """Import taxol data by condition"""
    print("untreated")
    untreated = [import_taxol_file("HC00801_A1_field_1_level_1.csv") +
    import_taxol_file("HC00801_A1_field_2_level_1.csv") +
    import_taxol_file("HC00801_A1_field_3_level_1.csv") +
    import_taxol_file("HC00801_A1_field_4_level_1.csv")]

    print("taxol 0.5")
    taxol_05 = [import_taxol_file("HC00801_A2_field_1_level_1.csv") +
    import_taxol_file("HC00801_A2_field_2_level_1.csv") +
    import_taxol_file("HC00801_A2_field_3_level_1.csv") +
    import_taxol_file("HC00801_A2_field_4_level_1.csv")]

    print("taxol 1")
    taxol_1 = [import_taxol_file("HC00801_B1_field_1_level_1.csv") +
    import_taxol_file("HC00801_B1_field_2_level_1.csv") +
    import_taxol_file("HC00801_B1_field_3_level_1.csv") +
    import_taxol_file("HC00801_B1_field_4_level_1.csv")]

    print("taxol 1.5")
    taxol_15 = [import_taxol_file("HC00801_B2_field_1_level_1.csv") +
    import_taxol_file("HC00801_B2_field_2_level_1.csv") +
    import_taxol_file("HC00801_B2_field_3_level_1.csv") +
    import_taxol_file("HC00801_B2_field_4_level_1.csv")]

    print("taxol 2")
    taxol_2 = [import_taxol_file("HC00801_C1_field_1_level_1.csv") +
    import_taxol_file("HC00801_C1_field_2_level_1.csv") +
    import_taxol_file("HC00801_C1_field_3_level_1.csv") +
    import_taxol_file("HC00801_C1_field_4_level_1.csv")]

    print("taxol 2.5")
    taxol_25 = [import_taxol_file("HC00801_C2_field_1_level_1.csv") +
    import_taxol_file("HC00801_C2_field_2_level_1.csv") +
    import_taxol_file("HC00801_C2_field_3_level_1.csv") +
    import_taxol_file("HC00801_C2_field_4_level_1.csv")]

    print("taxol 3")
    taxol_3 = [import_taxol_file("HC00801_D1_field_1_level_1.csv") +
    import_taxol_file("HC00801_D1_field_2_level_1.csv") +
    import_taxol_file("HC00801_D1_field_3_level_1.csv") +
    import_taxol_file("HC00801_D1_field_4_level_1.csv")]

    print("taxol 4")
    taxol_4 = [import_taxol_file("HC00801_D2_field_1_level_1.csv") +
    import_taxol_file("HC00801_D2_field_2_level_1.csv") +
    import_taxol_file("HC00801_D2_field_3_level_1.csv") +
    import_taxol_file("HC00801_D2_field_4_level_1.csv")]

    return untreated, taxol_05, taxol_1, taxol_15, taxol_2, taxol_25, taxol_3, taxol_4
