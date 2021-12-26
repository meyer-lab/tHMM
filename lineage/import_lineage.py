""" This file includes functions to import the new lineage data. """
import pandas as pd
import numpy as np
from .CellVar import CellVar as c

def read_lineage_data():
    """ Reading the data and extracting lineages with their corresponding observations. """
    df = pd.read_csv(r"lineage/data/LineageData/AU02101_A3_field_1_RP_50_CSV-Table.csv")

    population = []
    for i in range(2, np.max(data["lineageId"])+1): # loop over "lineageId"s, aka, each lineage
        lineage = df.loc[df['lineageId'] == i] # select all the cells that belong to that lineage

        if not(lineage.empty): # if the lineage Id exists, do the rest, if not, pass
            unique_trackIDs = lineage["trackId"].unique() # the length of this shows the number of cells in this lineage
            parent_trackIDs = lineage["parentTrackId"].unique() # self-explanatory!

            # the root parent cell
            parent_cell = c(parent=None, gen=1)
            parent_cell.obs = [0, 0, 0, 0] # [fate, lifetime, diameter, censored?]
            parent_cell.obs[1] = 0.5*(np.max(lineage.loc[lineage['trackId'] == unique_trackIDs[0]]['frame']) - np.min(lineage.loc[lineage['trackId'] == unique_trackIDs[0]]['frame']))
            parent_cell.obs[2] = np.mean(lineage.loc[lineage['trackId'] == unique_trackIDs[0]]['Diameter_0'])
            parent_cell.obs[3] = 1 # certainly left-censored.

            lineage.drop(lineage.loc[lineage['trackId'] == unique_trackIDs[0]].index, inplace=True)
            ### we can add any other feature here...
            lineage_list = [parent_cell]
            while not(lineage.empty):
                daughter.left, lineage = recurssive(parent_cell, lineage)
                daughter.right, lineage = recurssive(parent_cell, lineage)
                lineage_list.append(daughter.left)
                lineage_list.append(daughter.right)
                parent_cell = daughter

        population.append(lineage_list)
        
    return population

def recurssive(parent_cell, lineage):

    unique_trackIDs = lineage["trackId"].unique()
    parent_trackIDs = lineage["parentTrackId"].unique()

    if unique_trackIDs[0] in parent_trackIDs: # the cell of interest has divided
        daughter = c(parent=parent_cell, gen=parent_cell.gen+1)

        if len(unique_trackIDs) > 0: # only one of the daughters exist
            # left daughter
            daughter.left.obs = [0, 0, 0, 0]
            daughter.left.obs[0] = 0 #? I don't know yet. If 0 means died, if 1 means divided, if nan means we don't know
            daughter.left.obs[1] = 0.5*(np.max(lineage.loc[lineage['trackId'] == unique_trackIDs[0]]['frame']) - np.min(lineage.loc[lineage['trackId'] == unique_trackIDs[0]]['frame']))
            daughter.left.obs[2] = np.mean(lineage.loc[lineage['trackId'] == unique_trackIDs[0]]['Diameter_0'])
            daughter.left.obs[3] = 1 # TO BE DETERMINED
            # remove the parent cell from the lineage and try recurssion
            lineage.drop(lineage.loc[lineage['trackId'] == unique_trackIDs[0]].index, inplace=True)

        if len(unique_trackIDs) > 1: # both daughters exist
            # right daughter
            daughter.right.obs = [0, 0, 0, 0]
            daughter.right.obs[0] = 0 #? I don't know yet. If 0 means died, if 1 means divided, if nan means we don't know
            daughter.right.obs[1] = 0.5*(np.max(lineage.loc[lineage['trackId'] == unique_trackIDs[1]]['frame']) - np.min(lineage.loc[lineage['trackId'] == unique_trackIDs[1]]['frame']))
            daughter.right.obs[2] = np.mean(lineage.loc[lineage['trackId'] == unique_trackIDs[1]]['Diameter_0'])
            daughter.right.obs[3] = 1 # TO BE DETERMINED
            lineage.drop(lineage.loc[lineage['trackId'] == unique_trackIDs[1]].index, inplace=True)
        else:
            daughter.right = None

    else: # the cell of interest has not divided
        return # nada

    return daughter, lineage