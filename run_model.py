""" This file serves to import the cellular data from a CSV, convert it into the appropriate data types, and then run the tHMM on said data. To run this file from terminal for a given CSV file, use the command "python3 run_model.py data/[FILENAME].csv" """
import sys
from lineage.parse_csv import load_data, process

data = load_data(sys.argv[1]) # i.e. 'data/2019_02_21_true.csv'
pop1 = process(data, 5.0) # create a population of cells given the data and the imaging frequency
print("length of pop: " + str(len(pop1)))
for cell in pop1:
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
    print("\n")

