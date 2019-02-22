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
reps = 3
numStates = 2

lin_length = []

accuracy_holder_holder = []
bern = []
c = []
scale = []
cells = []


accuracy_st = []
bern_st = []
c_st = []
scale_st = []
cells_st = []

time1 = 80
time2 = 82
times = range(time1,time2)

for experimentTime in times:
    accuracy_holder = []
    bern_holder = []
    c_holder = []
    scale_holder = []
    cells_holder = []
    for rep in range(reps):
        X1 = gpt(experimentTime, initCells, locBern, cGom, scaleGom)
        X = remove_NaNs(X1)
        tHMMobj = tHMM(X, numStates=numStates) # build the tHMM class with X
        fit(tHMMobj, verbose=False)
        diag = np.diagonal(tHMMobj.paramlist[0]["T"])
        chosen_state = np.argmax(diag)
            
        deltas, state_ptrs = get_leaf_deltas(tHMMobj) # gets the deltas matrix
        get_nonleaf_deltas(tHMMobj, deltas, state_ptrs)
        v = Viterbi(tHMMobj, deltas, state_ptrs)

        counts = np.bincount(v[0])
        maxcount = np.argmax(counts)
        a = np.zeros((len(v[0])),dtype=int)
        for i in range(len(v[0])):
            a[i] = maxcount
        
        #write a holder for all lineages
        wrong = 0
        for num in range(tHMMobj.numLineages): # for each lineage in our Population
            lineage = tHMMobj.population[num] # getting the lineage in the Population by lineage index

            for cell in range(len(lineage)): # for each cell in the lineage
                if v[0][cell] == a[cell]:
                    pass
                    #print(viterbi[cell.index],actual[cell.index])
                else:
                    wrong = wrong + 1
            
        accuracy = (len(lineage) - wrong)/len(lineage) #must be fixed for more than 1 lineage
        
        accuracy_holder.append(accuracy)
        bern_holder.append(tHMMobj.paramlist[0]["E"][chosen_state,0])
        c_holder.append(tHMMobj.paramlist[0]["E"][chosen_state,1])
        scale_holder.append(tHMMobj.paramlist[0]["E"][chosen_state,2])
        cells_holder.append(len(lineage))
        
    accuracy_holder_holder.append(np.mean(accuracy_holder))
    bern.append(np.mean(bern_holder))
    c.append(np.mean(c_holder))
    scale.append(np.mean(scale_holder))
    cells.append(np.mean(cells_holder))

    accuracy_st.append(np.std(accuracy_holder))
    bern_st.append(np.std(bern_holder))
    c_st.append(np.std(c_holder))
    scale_st.append(np.std(scale_holder))
    cells_st.append(np.std(cells_holder))
        

x=cells

#figure 1a - accuracy




# Now switch to a more OO interface to exercise more features.
fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
ax = axs[0,0]
ax.set_xscale('log')
ax.errorbar(x, accuracy_holder_holder, yerr=accuracy_st, fmt='o', c='b',marker="^",label='accuracy',fillstyle='none')
ax.axhline(y=1, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='r',)

plt.legend(loc=2)
ax.set_title('Accuracy')
#ax.locator_params(nbins=4)
ax = axs[0,1]
ax.set_xscale('log')
ax.errorbar(x, bern, yerr=bern_st, fmt='o', c='b',marker="^",label='bern',fillstyle='none')
ax.set_title('Bernoulli Parameter')
ax.axhline(y=locBern, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='r',)

ax = axs[1,0]
ax.set_xscale('log')
ax.errorbar(x,c,yerr = c_st,fmt='o',c='g',marker=(8,2,0),label='c')
ax.axhline(y=cGom, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='r',)
ax.set_title('Gompertz C')
plt.xlabel('Average Number of Cells per Lineage')

ax = axs[1,1]
ax.set_xscale('log')
ax.errorbar(x,scale,yerr = scale_st,fmt='o',c='k',label='scale')
ax.axhline(y=scaleGom, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='r',)
ax.set_title('Gompertz Scale')

fig.suptitle('Variable errorbars')
plt.xlabel('Average Number of Cells per Lineage')
plt.savefig('test.png')

'''fig, ax = plt.subplots(2,2)

ax=fig.add_subplot(111)
plt.errorbar(x, accuracy_holder_holder, yerr=np.std(bern_st), fmt='o', c='b',marker="^",label='accuracy',fillstyle='none')
plt.axhline(y=1, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='r',)
plt.xscale('log')
plt.xlabel('Average Number of Cells per Lineage')
plt.legend(loc=2)
plt.savefig('Figure1a.png')

#figure 1b - bern


#fig2=plt.figure()
#ax=fig2.add_subplot(111)

fig2=plt.figure()
ax=fig.add_subplot(212)
plt.errorbar(x, bern, yerr=np.std(bern_st), fmt='o', c='b',marker="^",label='bern',fillstyle='none')
plt.axhline(y=locBern, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='r',)
plt.xscale('log')
plt.xlabel('Cells')
#ax.errorbar(lin_length,bern,yerr = np.std(bern),c='b',marker="^",ls='--',label='bern',fillstyle='none')

plt.legend(loc=2)
plt.savefig('Figure1b.png')

###figure 1c -gomp


fig3=plt.figure()
ax=fig3.add_subplot(321)


plt.errorbar(x,c,yerr = np.std(c_st),fmt='o',c='g',marker=(8,2,0),label='c')
plt.axhline(y=cGom, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='r',)
plt.xscale('log')
plt.xlabel('Cells')
plt.legend(loc=2)
plt.savefig('Figure1c.png')

#fig 4

fig4=plt.figure()
ax=fig4.add_subplot(422)


plt.errorbar(x,scale,yerr = np.std(scale_st),fmt='o',c='k',label='scale')
#plt.plot(y * points, linestyle=linestyle, color=color, linewidth=3)
plt.axhline(y=scaleGom, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='r',)
plt.xscale('log')
plt.xlabel('Cells')
plt.legend(loc=2)
plt.savefig('Figure1d.png')'''