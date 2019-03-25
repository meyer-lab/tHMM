""" This file serves to import the cellular data from a CSV, convert it into the appropriate data types, and then run the tHMM on said data. To run this file from terminal for a given CSV file, use the command "python3 run_model.py data/[FILENAME].csv" """
import sys
from lineage.parse_csv import load_data, process
from lineage.tHMM import tHMM
from lineage.BaumWelch import fit
from lineage.Viterbi import get_leaf_deltas, get_nonleaf_deltas, Viterbi
from lineage.plotting_utils import plot_experiments, plot_population

data = load_data(sys.argv[1]) # i.e. 'data/2019_02_21_true.csv'
print("true length of pop:", data.shape[0])
X = process(data, 1/6) # create a population of cells given the data and the imaging frequency
print("found pop length:", len(X))

numStates = 2
tHMMobj = tHMM(X, numStates=numStates, FOM='G') # build the tHMM class with X
fdir = './Experiment_Figs/'

print("number of lineages =", len(tHMMobj.population))
for ii, lin in enumerate(tHMMobj.population):
    plot_experiments(lin, fdir+"lineage"+str(ii+1)+".png")

tHMMobj, NF, betas, gammas = fit(tHMMobj, max_iter=100, verbose=False)

deltas, state_ptrs = get_leaf_deltas(tHMMobj) # gets the deltas matrix
get_nonleaf_deltas(tHMMobj, deltas, state_ptrs)
all_states = Viterbi(tHMMobj, deltas, state_ptrs)
print(all_states)

