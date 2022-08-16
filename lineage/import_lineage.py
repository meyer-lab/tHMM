""" This file includes functions to import the new lineage data. """
import pandas as pd
from collections import OrderedDict
from operator import itemgetter
import numpy as np
from .CellVar import CellVar

############################
# importing AU565 data (new)
############################
# path = "lineage/data/LineageData/AU02101_A3_field_1_RP_50_CSV-Table.csv"


def import_AU565(path: str) -> list[list[CellVar]]:
    """ Importing AU565 file cells.
    :param path: the path to the file.
    :return population: list of cells structured in CellVar objects.
    """
    df = pd.read_csv(path)

    population = []
    # loop over "lineageId"s
    df = df.sort_values(by=['lineageId'])
    for _, lineage in df.groupby('lineageId'):
        # Setup a track -> parent dict where the first entry is the root
        linRel = lineage[["trackId", "parentTrackId"]].drop_duplicates()
        linRel = linRel.sort_values(by=['parentTrackId', 'trackId'])
        linRel = OrderedDict(zip(linRel.trackId, linRel.parentTrackId))
        linRel = OrderedDict(sorted(linRel.items(), key=itemgetter(1))) # sort by parent

        # create the root parent cell and assign obsrvations
        lineage_list: list[CellVar] = list()

        for cellID, parentID in linRel.items():
            if parentID == 0:
                lineage_list.append(assign_observs_AU565(None, lineage, cellID))
            else:
                parentIDX = list(linRel.keys()).index(parentID)
                a = assign_observs_AU565(lineage_list[parentIDX], lineage, cellID)
                lineage_list.append(a)
                if lineage_list[parentIDX].left is None:
                    lineage_list[parentIDX].left = a
                else:
                    assert lineage_list[parentIDX].right is None
                    lineage_list[parentIDX].right = a

        assert len(lineage_list) == len(linRel)
        # if both observations are zero, remove the cell
        for n, cell in enumerate(lineage_list):
            if (cell.obs[1] == 0 and cell.obs[2] == 0):  # type: ignore
                lineage_list.pop(n)

        population.append(lineage_list)
    return population


def assign_observs_AU565(parent, lineage: pd.DataFrame, uniq_id: int) -> CellVar:
    """Given a cell, the lineage, and the unique id of the cell, it assigns the observations of that cell, and returns it.
    :param cell: a CellVar object to be assigned observations.
    :param lineage: the lineage list of cells that the given cell is from.
    :param uniq_id: the id given to the cell from the experiment.
    """
    # initialize
    cell = CellVar(parent=parent)
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


def import_MCF10A(path: str) -> list[list[CellVar]]:
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

        parent_cell = assign_observs_MCF10A(None, lineage, lin_code)

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

            a = assign_observs_MCF10A(cell, lineage, child_id[0])
            b = assign_observs_MCF10A(cell, lineage, child_id[1])
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


def assign_observs_MCF10A(parent, lineage, uniq_id: int):
    """Given a cell, the lineage, and the unique id of the cell, it assigns the observations of that cell, and returns it.
    :param cell: a CellVar object to be assigned observations.
    :param lineage: the lineage list of cells that the given cell is from.
    :param uniq_id: the id given to the cell from the experiment.
    """
    # initialize
    cell = CellVar(parent=parent)
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


def MCF10A(condition: str) -> list[list[CellVar]]:
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
