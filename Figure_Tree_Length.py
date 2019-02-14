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

lin_length = []

time1 = 40
time2 = 100
times = range(time1,time2)

accuracy_holder = []

for num in times:
    
        experimentTime = num
        X1 = gpt(experimentTime, initCells, locBern, cGom, scaleGom)
        X = remove_NaNs(X1)
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
        v = Viterbi(tHMMobj, deltas, state_ptrs)
        viterbi.append(v)
        
        
        counts = np.bincount(v[0])
        maxcount = np.argmax(counts)
        a = np.zeros((len(v[0])),dtype=int)
        for i in range(len(v[0])):
            a[i] = maxcount
        actual.append(a)
        
        #write a holder for all lineages
        wrong = 0
        for num in range(tHMMobj.numLineages): # for each lineage in our Population
            lineage = tHMMobj.population[num] # getting the lineage in the Population by lineage index
            '''
            print(v)
            print(a)
            print(len(v[0]))
            print(range(len(lineage)))
            '''
            for cell in range(len(lineage)): # for each cell in the lineage
                if v[0][cell] == a[cell]:
                    pass
                    #print(viterbi[cell.index],actual[cell.index])
                else:
                    wrong = wrong + 1
            
        accuracy = (len(lineage) - wrong)/len(lineage) #must be fixed for more than 1 lineage
        
        accuracy_holder.append(accuracy)
        
        lin_length.append(len(lineage))
        
x=np.arange(time1,time2)



#figure 1a - accuracy


fig1=plt.figure()
ax=fig1.add_subplot(111)
plt.errorbar(lin_length, accuracy_holder, yerr=np.std(bern), fmt='o', c='b',marker="^",label='accuracy',fillstyle='none')
plt.xlabel('Cells')
plt.legend(loc=2)
plt.savefig('Figure1a.png')

#figure 1b - bern


#fig2=plt.figure()
#ax=fig2.add_subplot(111)

fig2=plt.figure()
ax=fig2.add_subplot(111)
plt.errorbar(lin_length, bern, yerr=np.std(bern), fmt='o', c='b',marker="^",label='bern',fillstyle='none')
plt.axhline(y=locBern, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='r',)
plt.xlabel('Cells')
#ax.errorbar(lin_length,bern,yerr = np.std(bern),c='b',marker="^",ls='--',label='bern',fillstyle='none')

plt.legend(loc=2)
plt.savefig('Figure1b.png')

###figure 1c -gomp


fig3=plt.figure()
ax=fig3.add_subplot(111)


plt.errorbar(lin_length,c,yerr = np.std(c),fmt='o',c='g',marker=(8,2,0),label='c')
plt.axhline(y=cGom, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='r',)
plt.xlabel('Cells')
plt.legend(loc=2)
plt.savefig('Figure1c.png')

#fig 4

fig4=plt.figure()
ax=fig4.add_subplot(111)


plt.errorbar(lin_length,scale,yerr = np.std(scale),fmt='o',c='k',label='scale')
#plt.plot(y * points, linestyle=linestyle, color=color, linewidth=3)
plt.axhline(y=scaleGom, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='r',)
plt.xlabel('Cells')
plt.legend(loc=2)
plt.savefig('Figure1d.png')
