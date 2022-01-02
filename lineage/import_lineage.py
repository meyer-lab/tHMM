""" This file includes functions to import the new lineage data. """
import pandas as pd
import itertools
import numpy as np
from .CellVar import CellVar as c

############################
# importing AU565 data (new)
############################
# path = "lineage/data/LineageData/AU02101_A3_field_1_RP_50_CSV-Table.csv"

def import_AU565(path):
    """ Importing AU565 file cells. """
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
            parent_ids = list(itertools.chain(*pid))

            # create the root parent cell and assign obsrvations
            parent_cell = c(parent=None, gen=1)
            parent_cell = assign_observs_AU565(parent_cell, lineage, unique_cell_ids[0])

            # create a list to store cells belonging to a lineage
            lineage_list = [parent_cell]
            for k, val in enumerate(unique_cell_ids):
                if val in parent_ids:  # if the id of a cell exists in the parent ids, it means the cell divides
                    parent_index = [indx for indx, value in enumerate(parent_ids) if value == val]  # find whose mother it is
                    assert len(parent_index) == 2  # make sure has two children
                    lineage_list[k].left = c(parent=lineage_list[k], gen=lineage_list[k].gen + 1)
                    lineage_list[k].left = assign_observs_AU565(lineage_list[k].left, lineage, unique_cell_ids[parent_index[0]])
                    lineage_list[k].right = c(parent=lineage_list[k], gen=lineage_list[k].gen + 1)
                    lineage_list[k].right = assign_observs_AU565(lineage_list[k].right, lineage, unique_cell_ids[parent_index[1]])

                    lineage_list.append(lineage_list[k].left)
                    lineage_list.append(lineage_list[k].right)

        assert len(lineage_list) == len(unique_cell_ids)
        # if both observations are zero, remove the cell
        for n, cell in enumerate(lineage_list):
            if (cell.obs[1] == 0 and cell.obs[2] == 0):
                lineage_list.pop(n)

        # give all cells of the same lineage a track id
        for cell in lineage_list:
            cell.lineageID = i

        population.append(lineage_list)
    return population

def assign_observs_AU565(cell, lineage: list, uniq_id: int):
    """Given a cell, the lineage, and the unique id of the cell, it assigns the observations of that cell, and returns it."""
    # initialize
    cell.obs = [1, 0, 0, 0]
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
        if diam.size == 0:  # if all are nan
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

def import_MCF10A(path):
    """ Reading the data and extracting lineages and assigning their corresponding observations. """

    population = []
    # loop over "lineageId"s
    for i in df["lineage"].unique():
        # select all the cells that belong to that lineage
        lineage = df.loc[df['lineage'] == i]

        unique_cell_ids = list(lineage["TID"].unique())  # the length of this shows the number of cells in this lineage
        unique_parent_trackIDs = lineage["motherID"].unique()

        # if a cell has 3 children, remove the third one.
        cell_id = []
        for j in unique_parent_trackIDs:
            tmp = lineage.loc[lineage["motherID"]==j]
            trackIDs = list(tmp["TID"].unique())
            cell_id.append(trackIDs[0:2])

        # make sure there is still at least one cell in the lineage
        if not(lineage.empty):
            pid = [[0]]  # root parent's parent id
            for j in unique_parent_trackIDs:
                if j != 0:
                    pid.append(2 * [j])
            parent_ids = list(itertools.chain(*pid))
            unique_cell_ids = list(itertools.chain(*cell_id))

            if len(parent_ids) != len(unique_cell_ids):
                continue

            parent_cell = c(parent=None, gen=1)
            parent_cell = assign_observs_MCF10A(parent_cell, lineage, unique_cell_ids[0])

            # create a list to store cells belonging to a lineage
            lineage_list = [parent_cell]
            for k, val in enumerate(unique_cell_ids):
                if val in parent_ids: # if the id of a cell exists in the parent ids, it means the cell divides
                    parent_index = [indx for indx, value in enumerate(parent_ids) if value == val] # find whose mother it is
                    assert len(parent_index) == 2 # make sure has two children
                    lineage_list[k].left = c(parent=lineage_list[k], gen=lineage_list[k].gen+1)
                    lineage_list[k].left = assign_observs_MCF10A(lineage_list[k].left, lineage, unique_cell_ids[parent_index[0]])
                    lineage_list[k].right = c(parent=lineage_list[k], gen=lineage_list[k].gen+1)
                    lineage_list[k].right = assign_observs_MCF10A(lineage_list[k].right, lineage, unique_cell_ids[parent_index[1]])

                    lineage_list.append(lineage_list[k].left)
                    lineage_list.append(lineage_list[k].right)

        else:
            break

        assert len(lineage_list) == len(unique_cell_ids)
        # if both observations are zero, remove the cell
        for n, cell in enumerate(lineage_list):
            if (cell.obs[1] == 0 and cell.obs[2] == 0):
                lineage_list.pop(n)

        # give all cells of the same lineage a track id
        for cell in lineage_list:
            cell.lineageID = i

        population.append(lineage_list)
    return population

def assign_observs_MCF10A(cell, lineage: list, uniq_id: int):
    """Given a cell, the lineage, and the unique id of the cell, it assigns the observations of that cell, and returns it."""
    # initialize
    cell.obs = [1, 0, 0] # [fate, lifetime, censored?]
    parent_id = lineage["motherID"].unique()
    # cell fate: die = 0, divide = 1
    if not(uniq_id in parent_id): # if the cell has not divided, means either died or reached experiment end time
        if np.max(lineage.loc[lineage["TID"] == uniq_id]["tmin"]) == 2880: # means reached end of experiment
            cell.obs[0] = np.nan # don't know
            cell.obs[2] = 1 # censored
        else: # means cell died before experiment ended
            cell.obs[0] = 0 # died
            cell.obs[2] = 0 # not censored

    if cell.gen == 1: # it is root parent
        cell.obs[2] = 1 # meaning it is left censored in its lifetime

    # cell's lifetime
    cell.obs[1] = (np.max(lineage.loc[lineage['TID'] == uniq_id]['tmin']) - np.min(lineage.loc[lineage['TID'] == uniq_id]['tmin'])) / 60

    return cell

def MCF10A(condition:str):
    """ Creates the population of lineages for each condition.
    Conditions include: PBS, EGF-treated, HGF-treated, OSM-treated. """
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
        # data2 = import_MCF10A("lineage/data/MCF10A/HGF_2.csv")
        data3 = import_MCF10A("lineage/data/MCF10A/HGF_3.csv")
        data4 = import_MCF10A("lineage/data/MCF10A/HGF_4.csv")
        data5 = import_MCF10A("lineage/data/MCF10A/HGF_5.csv")
        return data1 + data3 + data4 + data5

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
