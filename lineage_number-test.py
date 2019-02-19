import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')
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

experimentTime = 60
locBern = [0.999]
cGom = [2]
scaleGom = [40]
reps = 3
numStates = 2
bern = []
c = []
scale = []


accuracy_st = []
bern_st = []
c_st = []
scale_st = []


length_holder_tHMM = []

lineage1 = 1
lineage2 = 8
populations = range(lineage1,lineage2)

accuracy_holder_tHMM = [] #list of lists of lists

for num in populations: #a pop with num number of lineages
    accuracy_holder_lin = [] #list of lists        
    bern_holder = []
    c_holder = []
    scale_holder = []  
    for rep in range(reps):
        accuracy_rep = []
        bern_rep = []
        c_rep = []
        scale_rep = []
        
        initCells = [num]
        X1 = gpt(experimentTime, initCells, locBern, cGom, scaleGom)
        X = remove_NaNs(X1)
        tHMMobj = tHMM(X, numStates=numStates) # build the tHMM class with X
        fit(tHMMobj, verbose=False)
        for lin in range(tHMMobj.numLineages): #go through each lineage for a thmm with a set of lineages
            diag = np.diagonal(tHMMobj.paramlist[lin]["T"])
            chosen_state = np.argmax(diag)
            deltas, state_ptrs = get_leaf_deltas(tHMMobj) # gets the deltas matrix
            get_nonleaf_deltas(tHMMobj, deltas, state_ptrs)
            v = Viterbi(tHMMobj, deltas, state_ptrs)
            counts = np.bincount(v[lin])
            maxcount = np.argmax(counts)
            a = np.zeros((len(v[lin])),dtype=int)
            for i in range(len(v[lin])):
                a[i] = maxcount
            wrong = 0
            lineage = tHMMobj.population[lin] # getting the lineage in the Population by lineage index
            for cell in range(len(lineage)): # for each cell in the lineage
                if v[lin][cell] == a[cell]:
                    pass
                            #print(viterbi[cell.index],actual[cell.index])
                else:
                    wrong = wrong + 1

            accuracy = (len(lineage) - wrong)/len(lineage) #must be fixed for more than 1 lineage
            
            accuracy_rep.append(accuracy)
            bern_rep.append(tHMMobj.paramlist[lin]["E"][chosen_state,0])
            c_rep.append(tHMMobj.paramlist[lin]["E"][chosen_state,1])
            scale_rep.append(tHMMobj.paramlist[lin]["E"][chosen_state,2])
        
        accuracy_holder_lin.append(np.mean(accuracy_rep))
        bern_holder.append(np.mean(bern_rep))
        c_holder.append(np.mean(c_rep))
        scale_holder.append(np.mean(scale_rep))
    
    accuracy_holder_tHMM.append(np.mean(accuracy_holder_lin))
    length_holder_tHMM.append(num)
    bern.append(np.mean(bern_holder))
    c.append(np.mean(c_holder))
    scale.append(np.mean(scale_holder))
    
    accuracy_st.append(np.std(accuracy_holder_lin))
    bern_st.append(np.std(bern_holder))
    c_st.append(np.std(c_holder))
    scale_st.append(np.std(scale_holder))
    
    x=length_holder_tHMM

fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
ax = axs[0,0]
plt.xlabel('Average Number of Cells per Lineage')
ax.errorbar(x, accuracy_holder_tHMM, yerr=accuracy_st, fmt='o', c='b',marker="*",fillstyle='none')
ax.axhline(y=1, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='r',)

ax.set_title('Accuracy')
#ax.locator_params(nbins=4)
ax = axs[0,1]
plt.xlabel('Average Number of Cells per Lineage')
ax.errorbar(x, bern, yerr=bern_st, fmt='o', c='b',marker="^",fillstyle='none')
ax.set_title('Bernoulli Parameter')
ax.axhline(y=locBern, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='r',)

ax = axs[1,0]
ax.set_xlabel('Lineages per Population')
ax.errorbar(x,c,yerr = c_st,fmt='o',c='g',marker=(8,2,0))
ax.axhline(y=cGom, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='r',)
ax.set_title('Gompertz C')


ax = axs[1,1]
ax.set_xlabel('Lineages per Population')
ax.errorbar(x,scale,yerr = scale_st,fmt='o',c='k')
ax.axhline(y=scaleGom, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='r',)
ax.set_title('Gompertz Scale')

fig.suptitle('Lineage Length effect on tHMM Classification')

plt.savefig('lineage_number_classification.png')