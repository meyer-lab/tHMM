from os.path import join, dirname, abspath
import pandas as pd
import numpy as np
from .CellNode import CellNode

def load_data(filename):
    """ Return path of CSV files. """
    path = dirname(abspath(__file__))
    data = pd.read_csv(join(path, filename)).values
    # convert away from strings
    ii = 0
    while ii < data.shape[0]:
        for jj in range(data.shape[1]):
            if data[ii,jj] == "None": # if entry is the string None
                data[ii,jj] = None # reassign entry to keyword None
            else:
                data[ii,jj] = float(data[ii,jj]) # force all strings into floats
        # delete rows for cells that are seen in the first frame
        if data[ii, 1] == 0:
            data = np.delete(data, ii, 0) # delete row
        else:
            ii += 1 # increment by 1 if you don't delete the row

    return data

def process(data, deltaT):
    """ Convert the CSV file into population of cells with each cell being a CellNode object. deltaT represents the time lapse between images. """
    pop = []
    # search for root nodes and add them to the population
    linID = 0
    for ii in range(data.shape[0]):
        if is_root_node(data, ii): # if this row represents a root node
            pop.append(CellNode(linID=linID, trackID=data[ii, 0], startT=(deltaT*data[ii, 1]), true_state=data[ii, 7]))
            linID += 1 # increment lineage by 1 for each root node added

    # cycle through to handle divisions and deaths using CellNode functions
    for cell in pop:
        row = find_row(data, cell) # find the row of said cell in the CSV file
        if data[row, 3] == 0: # if the cell dies
            cell.die(data[row, 2]*deltaT) # force it to die at the last frame
        if data[row, 3] == 1: # if the cell divides
            temp1, temp2 = cell.divide(endT=data[row, 2]*deltaT, trackID_d1=data[row, 5], trackID_d2=data[row, 6])
            pop.append(temp1)
            pop.append(temp2)

    return pop

def is_root_node(data, row):
    """ Returns True if none the object_ID (0 position) is not found in either of the child columns for any other cell"""
    temp = True
    obj_ID = data[row, 0] # store the object ID for the cell of interest
    for ii in range(data.shape[0]):
        if data[ii, 5] is not None: # can only make integer comparison if not None
            if data[ii, 5] == obj_ID: # if object ID is found to be one of the daughter cells
                temp = False
                break
        if data[ii, 6] is not None: # can only make integer comparison if not None
            if data[ii, 6] == obj_ID: # if object ID is found to be one of the daughter cells
                temp = False
                break
    return temp

def find_row(data, cell):
    """ Returns the row number of the data matrix that corresponds to the CellNode (cell) given. """
    ID = cell.trackID
    row = -1
    for ii in range(0, data.shape[0]):
        if ID == data[ii, 0]: # if the ID matches the first column of said row
            row = ii
            break
    assert row >= 0 # make sure the row was actually found
    return row
