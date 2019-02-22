from os.path import join, dirname, abspath
import pandas as pd
import numpy as np

def load_data(filename):
    """ Return path of CSV files. """
    path = dirname(abspath(__file__))
    return pd.read_csv(join(path, filename)).values

data = load_data('../data/Tracking_w_Learning.csv')
data = data[:, 0:6] # only need the first 5 columns of data
print(data.shape)

def process(data):
    """ Convert the CSV file into population of cells with each cell being a CellNode object. """
