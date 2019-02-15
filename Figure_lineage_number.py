'''Makes Figure 1 of Tree Length vs. Predictive Accuracy'''
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



experimentTime = 90
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
lineage2 = 3
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

            
x=np.arange(lineage1,lineage2)



























'''#figure 1a - accuracy


fig1=plt.figure()
ax=fig1.add_subplot(111)
plt.errorbar(length_holder_tHMM, accuracy_holder_tHMM, yerr=accuracy_st, fmt='o', c='b',marker="^",label='accuracy',fillstyle='none')
plt.axhline(y=1, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='r',)
plt.xscale('log')
plt.xlabel('Cells')
plt.legend(loc=2)
plt.savefig('Figure2a.png')

#figure 1b - bern


#fig2=plt.figure()
#ax=fig2.add_subplot(111)

fig2=plt.figure()
ax=fig2.add_subplot(111)
plt.errorbar(length_holder_tHMM, bern, yerr=bern_st, fmt='o', c='b',marker="^",label='bern',fillstyle='none')
plt.axhline(y=locBern, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='r',)
plt.xscale('log')
plt.xlabel('Cells')
#ax.errorbar(lin_length,bern,yerr = np.std(bern),c='b',marker="^",ls='--',label='bern',fillstyle='none')

plt.legend(loc=2)
plt.savefig('Figure2b.png')

###figure 1c -gomp


fig3=plt.figure()
ax=fig3.add_subplot(111)


plt.errorbar(length_holder_tHMM,c,yerr = c_st,fmt='o',c='g',marker=(8,2,0),label='c')
plt.axhline(y=cGom, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='r',)
plt.xscale('log')
plt.xlabel('Cells')
plt.legend(loc=2)
plt.savefig('Figure2c.png')

#fig 4

fig4=plt.figure()
ax=fig4.add_subplot(111)


plt.errorbar(length_holder_tHMM,scale,yerr = scale_st,fmt='o',c='k',label='scale')
#plt.plot(y * points, linestyle=linestyle, color=color, linewidth=3)
plt.axhline(y=scaleGom, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='r',)
plt.xscale('log')
plt.xlabel('Cells')
plt.legend(loc=2)
plt.savefig('Figure2d.png')
'''