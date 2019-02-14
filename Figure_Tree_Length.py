'''Makes Figure 1 of Tree Length vs. Predictive Accuracy'''
import unittest
import numpy as np
from matplotlib import pyplot as plt

from lineage.BaumWelch import fit
from lineage.DownwardRecursion import get_root_gammas, get_nonroot_gammas
from lineage.Viterbi import get_leaf_deltas, get_nonleaf_deltas, Viterbi
from lineage.UpwardRecursion import get_leaf_Normalizing_Factors, get_leaf_betas, get_nonleaf_NF_and_betas
from lineage.tHMM import tHMM
from lineage.tHMM_utils import max_gen, get_gen, get_parents_for_level
from lineage.Lineage_utils import remove_NaNs, get_numLineages, init_Population
from lineage.Lineage_utils import generatePopulationWithTime as gpt
from lineage.CellNode import CellNode


experimentTime = 50
initCells = [1] # there should be around 50 lineages b/c there are 50 initial cells
locBern = [0.999]
cGom = [2]
scaleGom = [40]
reps = 10
numStates = 2
bern = []
c = []
scale = []
actual = []
viterbi = []

times = range(10,101)

accuracy_holder = []

for num in times:
    
        experimentTime = num
        X = gpt(experimentTime, initCells, locBern, cGom, scaleGom)
        X = remove_NaNs(X)
        tHMMobj = tHMM(X, numStates=numStates) # build the tHMM class with X
        fit(tHMMobj, verbose=False)
            
        diag = np.diagonal(tHMMobj.paramlist[0]["T"])
        chosen_state = np.argmax(diag)
            
        bern.append(tHMMobj.paramlist[0]["E"][chosen_state,0])
        #print("\nRun {} Bernoulli p: {}".format(num, bern[num]))
        c.append(tHMMobj.paramlist[0]["E"][chosen_state,1])
        #print("Run {} Gompertz c: {}".format(num, c[num]))
        scale.append(tHMMobj.paramlist[0]["E"][chosen_state,2])
        #print("Run {} Gompertz scale: {}".format(num, scale[num]))
        #print("Run {} Initial Probabilities: ".format(num))
        #print(tHMMobj.paramlist[0]["pi"])
        #print("Run {} Transition Matrix: ".format(num))
        #print(tHMMobj.paramlist[0]["T"])
        deltas, state_ptrs = get_leaf_deltas(tHMMobj) # gets the deltas matrix
        get_nonleaf_deltas(tHMMobj, deltas, state_ptrs)
        viterbi.append(Viterbi(tHMMobj, deltas, state_ptrs))
        actual.append(np.zeros((num),dtype=int))
        
        #write a holder for all lineages
        wrong = 0
        for num in range(numLineages): # for each lineage in our Population
            lineage = population[num] # getting the lineage in the Population by lineage index
            for cell in lineage: # for each cell in the lineage
                if viterbi[num] == actual[num]:
                    pass
                else:
                    wrong = wrong + 1
            
        accuracy = (len(lineage) - wrong)/len(lineage)
        
        accuracy_holder.append(accuracy)
        
        
x=np.arange(10,101)

fig=plt.figure()
ax=fig.add_subplot(111)

ax.plot(x,accuracy_holder,c='b',marker="^",ls='--',label='accuracy',fillstyle='none')

plt.legend(loc=2)
plt.savefig('Figure1a.png')



fig=plt.figure()
ax=fig.add_subplot(111)

ax.errorbar(x,bern,yerr = np.std(bern),c='b',marker="^",ls='--',label='bern',fillstyle='none')
ax.errorbar(x,c,yerr = np.std(c),c='g',marker=(8,2,0),ls='--',label='c')
ax.errorbar(x,scale,yerr = np.std(scale),c='k',ls='-',label='scale')

plt.legend(loc=2)
plt.savefig('Figure1b.png')

