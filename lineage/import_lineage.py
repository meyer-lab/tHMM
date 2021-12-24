""" This file includes functions to import the new lineage data. """
import pandas as pd
import numpy as np

def read_lineage_data():
    """ Reading the data and extracting lineages with their corresponding observations. """
    df = pd.read_csv(r"lineage/data/LineageData/AU02101_A3_field_1_RP_50_CSV-Table.csv")

    population = []
    for i in range(2, np.max(data["lineageId"])+1): # loop over "lineageId"s, aka, each lineage
        lineage = df.loc[df['lineageId'] == i] # select all the cells that belong to that lineage

        if not(lineage.empty): # if the lineage Id exists, do the rest, if not, pass
            unique_trackIDs = lineage["trackId"].unique() # the length of this shows the number of cells in this lineage

            # the root parent cell
            parent_cell = c(parent=None, gen=1)
            parent_cell.obs = [0, 0, 0, 0] # [fate, lifetime, diameter, censored?]
            parent_cell.obs[1] = 0.5*(np.max(lin1.loc[lin1['trackId'] == unique_trackIDs[0]]['frame']) - np.min(lin1.loc[lin1['trackId'] == unique_trackIDs[0]]['frame']))
            parent_cell.obs[2] = np.mean(lin1.loc[lin1['trackId'] == unique_trackIDs[0]]['Diameter_0'])
            parent_cell.obs[3] = 1 # certainly left-censored.

            ### we can add any other feature here...
            lineage_list = [parent_cell]
            for val in unique_trackIDs[1:]: # leave out the very first element which belongs to the mother cell
                daughter = c(parent=parent_cell, gen=parent_cell.gen+1)

                #CELL FATE: TO BE FIGURED OUT...
                daughter.obs = [0, 0, 0, 0]
                daughter.obs[0] = 0 #? I don't know yet. If 0 means died, if 1 means divided, if nan means we don't know
                daughter.obs[1] = 0.5*(np.max(lin1.loc[lin1['trackId'] == val]['frame']) - np.min(lin1.loc[lin1['trackId'] == val]['frame']))
                daughter.obs[2] = np.mean(lin1.loc[lin1['trackId'] == val]['Diameter_0'])

                if np.isnan(daughter.obs[0]):
                    # TO BE DETERMINED...
                    daughter.obs[3] = 1 # means censored, indeed!
                else:
                    daughter.obs[3] = 0 # means not censored.

                lineage_list.append(daughter)
                parent_cell = daughter
        else: # means the lineageId doesn't exist, so pass
            pass
        population.append(lineage_list)
        
    return population
