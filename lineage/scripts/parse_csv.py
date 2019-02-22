from os.path import join, dirname, abspath
import pandas as pd
import numpy as np
from ..CellNode import CellNode

def load_data(filename):
    """ Return path of CSV files. """
    path = dirname(abspath(__file__))
    return pd.read_csv(join(path, filename)).values

data = load_data('../data/capstone-practice-format.csv')
print(data.shape)

def process(data, deltaT):
    """ Convert the CSV file into population of cells with each cell being a CellNode object. deltaT represents the time lapse between images. """
    # first need to find all the root nodes
    #root_node_ids = [] 
    #for ii in range(data.shape[0]):
    #    
    #    for jj in range(data.shap[0]):
    # put all cells in pop but ignore linID and left/right pointers for now
    pop = []
    for ii in range(data.shape[0]):
        objID = data[ii, 0]
        startT = deltaT * data[ii, 1]
        fate = data[ii, 3]
        if fate == -1: # if cell is unfinished at end of imaging
            fate = None # assign fate back to None
            endT = None
        else: # if cell divides or dies we can assign the endT based on final frame
            endT = deltaT * data[ii, 2]
        pop.append(CellNode())

def is_root_node(data, row):
    """ Returns True if none the object_ID (0 position) is not found in either of the child columns for any other cell"""
    temp = True
    obj_ID = data[row, 0] # store the object ID for the cell of interest
    for ii in range(data.shape[0]):
        if data[row]