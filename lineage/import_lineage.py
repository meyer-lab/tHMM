""" This file includes functions to import the new lineage data. """
import pandas as pd
import itertools
from more_itertools import locate
import numpy as np
from .CellVar import CellVar as c

def read_lineage_data():
    """ Reading the data and extracting lineages with their corresponding observations. """
    df = pd.read_csv(r"lineage/data/LineageData/AU02101_A3_field_1_RP_50_CSV-Table.csv")

    population = []
    for i in range(2, np.max(data["lineageId"])+1): # loop over "lineageId"s, aka, each lineage
        lineage = df.loc[df['lineageId'] == i] # select all the cells that belong to that lineage

        if not(lineage.empty): # if the lineage Id exists, do the rest, if not, pass
            unique_cell_ids = list(lineage["trackId"].unique()) # the length of this shows the number of cells in this lineage
            unique_parent_trackIDs = lineage["parentTrackId"].unique() # self-explanatory!

            pid = [[0]] # root parnt's parent id
            for i in unique_parent_trackIDs:
                if i != 0:
                    pid.append(np.count_nonzero(lineage["parentTrackId"]== i) * [i])
            parent_ids = list(itertools.chain(*pid))

            # the root parent cell
            parent_cell = c(parent=None, gen=1)
            parent_cell.obs = [0, 0, 0, 0] # [fate, lifetime, diameter, censored?]
            parent_cell.obs[1] = 0.5*(np.max(lineage.loc[lineage['trackId'] == unique_cell_ids[0]]['frame']) - np.min(lineage.loc[lineage['trackId'] == unique_cell_ids[0]]['frame']))
            parent_cell.obs[2] = np.mean(lineage.loc[lineage['trackId'] == unique_cell_ids[0]]['Diameter_0'])
            parent_cell.obs[3] = 1 # certainly left-censored.

            lineage_list = [parent_cell]
            for i, val in enumerate(unique_cell_ids):
                if val in parent_ids: # divides
                    parent_index = [indx for indx, value in enumerate(parent_ids) if value == val] # find whose mother it is
                    if len(parent_index) > 1: # has two children
                        lineage_list[i].left = c(parent=lineage_list[i], gen=lineage_list[i].gen+1)
                        lineage_list[i].left = assign_observs(lineage_list[i].left, lineage, unique_cell_ids[parent_index[0]])
                        lineage_list[i].right = c(parent=lineage_list[i], gen=lineage_list[i].gen+1)
                        lineage_list[i].right = assign_observs(lineage_list[i].right, lineage, unique_cell_ids[parent_index[1]])
                    elif len(parent_index) == 1: # has one child
                        lineage_list[i].left = c(parent=lineage_list[i], gen=lineage_list[i].gen+1)
                        lineage_list[i].left = assign_observs(lineage_list[i].left, lineage, unique_cell_ids[parent_index[0]])
                        lineage_list[i].right = None
                    else:
                        print("error here!")
                    lineage_list.append(lineage_list[i].left)
                    lineage_list.append(lineage_list[i].right)

        population.append(lineage_list)
        
    return population

def assign_observs(cell, lineage, uniq_id):
    cell.obs = [0, 0, 0, 0]
    # cell.obs[0] = ?
    cell.obs[1] = 0.5*(np.max(lineage.loc[lineage['trackId'] == uniq_id]['frame']) - np.min(lineage.loc[lineage['trackId'] == uniq_id]['frame']))
    cell.obs[2] = np.mean(lineage.loc[lineage['trackId'] == uniq_id]['Diameter_0'])
    # cell.obs[3] = ?
    return cell
