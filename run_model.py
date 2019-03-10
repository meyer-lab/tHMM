""" This file serves to import the cellular data from a CSV, convert it into the appropriate data types, and then run the tHMM on said data. To run this file from terminal for a given CSV file, use the command "python3 run_model.py data/[FILENAME].csv" """
import sys
import numpy as np
from lineage.parse_csv import load_data, process

data = load_data(sys.argv[1]) # i.e. 'data/2019_02_21_true.csv'
print("true length of pop:", data.shape[0])
pop1 = process(data, 10.0) # create a population of cells given the data and the imaging frequency
print("found pop length:", len(pop1))
