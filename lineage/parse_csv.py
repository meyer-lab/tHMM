from os.path import join, dirname, abspath
import pandas as pd
import numpy as np
from CellNode import CellNode

def load_data(filename):
    """ Return path of CSV files. """
    path = dirname(abspath(__file__))
    return pd.read_csv(join(path, filename)).values

data = load_data('data/capstone-practice-format.csv')
print(data.shape)

def process(data, deltaT):
    """ Convert the CSV file into population of cells with each cell being a CellNode object. deltaT represents the time lapse between images. """
    pop = []
    # search for root nodes and add them to the population
    linID = 0
    for ii in range(data.shape[0]):
        #print("ii: " + str(ii))
        if is_root_node(data, ii): # if this row represents a root node
            #print("function thinks the " + str(ii) + " row is a root node")
            pop.append(CellNode(linID=linID, trackID=data[ii, 0], startT=(deltaT*data[ii, 1])))
            linID += 1 # increment by 1 for each root node added

    # cycle through to handle divisions and deaths using CellNode functions
    for cell in pop:
        # print("len(pop): " + str(len(pop)))
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
    obj_ID = int(data[row, 0]) # store the object ID for the cell of interest
    print("row: " + str(row))
    print("obj_ID: " + str(obj_ID))
    for ii in range(data.shape[0]):
        print("ii: " + str(ii))
        print("data[ii, 5]: " + str(data[ii, 5]))
        print("data[ii, 6]: " + str(data[ii, 6]))
        # default left and right to be integers not seen in the data
        left = -99
        right = -99
        if data[ii, 5] is not None:
            left = int(data[ii, 5])
        if data[ii, 6] is not None:
            right = int(data[ii, 6])
        if obj_ID == left or obj_ID == right: # if the objectID is found to be one of the daughter cells
            print("going into if statement")
            temp = False # data in this row doesn't represent a root node
            break
    return temp

def find_row(data, cell):
    """ Returns the row number of the data matrix that corresponds to the CellNode (cell) given. """
    ID = cell.trackID
    # print("ID: " + str(ID))
    row = -1
    for ii in range(0, data.shape[0]):
        #print("ii: " + str(ii))
        #print("data[ii, 0]: " + str(data[ii, 0]))
        if ID == data[ii, 0]: # if the ID matches the first column of said row
            # print("assigning row")
            row = ii
            break
    assert row >= 0 # make sure the row was actually found
    return row


pop1 = process(data, 5.0)
print(len(pop1))