""" This file serves to import the cellular data from a CSV, convert it into the appropriate data types, and then run the tHMM on said data. To run this file from terminal for a given CSV file, use the command "python3 run_model.py data/[FILENAME].csv" """
import sys
import numpy as np
from lineage.parse_csv import load_data, process

data = load_data(sys.argv[1]) # i.e. 'data/2019_02_21_true.csv'
print("true length of pop:", data.shape)
pop1 = process(data, 10.0) # create a population of cells given the data and the imaging frequency
found_IDs = list()
for cell in pop1:
    found_IDs.append(cell.trackID)
    '''
    print("\n")
    print("trackID: " + str(cell.trackID))
    print("linID: " + str(cell.linID))
    print("true_state: " + str(cell.true_state))
    print("gen: " + str(cell.gen))
    print("startT: " + str(cell.startT))
    print("endT: " + str(cell.endT))
    print("tau: " + str(cell.tau))
    print("fate: " + str(cell.fate))
    if cell.left is not None:
        print("left.trackID: " + str(cell.left.trackID))
    if cell.right is not None:
        print("right.trackID: " + str(cell.right.trackID))
    if cell.parent is not None:
        print("parent.trackID: " + str(cell.parent.trackID))
'''
print("length of pop found: " + str(len(pop1)))
num_mult = 0
true_num = data.shape[0] # the true number of cells
print(true_num)
num_IDs = np.zeros((true_num))
for ID in found_IDs:
    for ii in range(true_num):
        if ID == data[ii, 0]:
            num_IDs[ii] += 1
            if num_IDs[ii] >= 2:
                print("position", ii, "has multple matches")
                print("cell ID:", data[ii, 0])
                num_mult += 1

print(num_mult)
