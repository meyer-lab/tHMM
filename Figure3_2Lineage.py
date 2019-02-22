import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.stats import entropy


from lineage.BaumWelch import fit
from lineage.DownwardRecursion import get_root_gammas, get_nonroot_gammas
from lineage.Viterbi import get_leaf_deltas, get_nonleaf_deltas, Viterbi
from lineage.UpwardRecursion import get_leaf_Normalizing_Factors, get_leaf_betas, get_nonleaf_NF_and_betas
from lineage.tHMM import tHMM
from lineage.tHMM_utils import max_gen, get_gen, get_parents_for_level
from lineage.Lineage_utils import remove_NaNs, get_numLineages, init_Population
from lineage.Lineage_utils import generatePopulationWithTime as gpt
from lineage.CellNode import CellNode



T_MAS = 200
T_2 = 75
times = range(1,2) 
reps = 3

MASinitCells = [1]
MASlocBern = [0.7]
MAScGom = [0.5]
MASscaleGom = [30]
initCells2 = [1]
locBern2 = [0.7]
cGom2 = [.5]
scaleGom2 = [30]

acc_h1 = [] #list of lists of lists
cell_h1 = []
bern_MAS_h1 = []
bern_2_h1 = []
cGom_MAS_h1 = []
cGom_2_h1 = []
scaleGom_MAS_h1 = []
scaleGom_2_h1 = []
EL_MAS_h1 = []
EL_2_h1 = []
KL_h1 = []

for experimentTime in times: #a pop with num number of lineages
    
    acc_h2 = []
    cell_h2 = []
    bern_h2 = []
    bern_MAS_h2 = []
    bern_2_h2 = []
    cGom_MAS_h2 = []
    cGom_2_h2 = []
    scaleGom_MAS_h2 = []
    scaleGom_2_h2 = []
    EL_MAS_h2 = []
    EL_2_h2 = []
    KL_h2 = []
    
    for rep in range(reps):
        print('Rep:', rep)
        MASexperimentTime = T_MAS
        masterLineage = gpt(MASexperimentTime, MASinitCells, MASlocBern, MAScGom, MASscaleGom)
        masterLineage = remove_NaNs(masterLineage)
        while len(masterLineage) == 0:
            masterLineage = gpt(MASexperimentTime, MASinitCells, MASlocBern, MAScGom, MASscaleGom)
            masterLineage = remove_NaNs(masterLineage)
        experimentTime2 = T_2
        sublineage2 = gpt(experimentTime2, initCells2, locBern2, cGom2, scaleGom2)
        sublineage2 = remove_NaNs(sublineage2)
        while len(sublineage2) == 0:
            sublineage2 = gpt(experimentTime2, initCells2, locBern2, cGom2, scaleGom2)
            sublineage2 = remove_NaNs(sublineage2)

        cell_endT_holder = []
        for cell in masterLineage:
            cell_endT_holder.append(cell.endT)

        master_cell_endT = max(cell_endT_holder) # get the longest tau in the list
        master_cell_endT_idx = np.argmax(cell_endT_holder) # get the idx of the longest tau in the lineage
        master_cell = masterLineage[master_cell_endT_idx] # get the master cell via the longest tau index

        for cell in sublineage2:
            cell.linID = master_cell.linID
            cell.gen += master_cell.gen
            cell.startT += master_cell_endT
            cell.endT += master_cell_endT

        master_cell.left = sublineage2[0]
        sublineage2[0].parent = master_cell
        newLineage = masterLineage + sublineage2
        
        
        X = remove_NaNs(newLineage)
        print(len(newLineage))
        numStates = 2
        tHMMobj = tHMM(X, numStates=numStates) # build the tHMM class with X
        fit(tHMMobj, max_iter=200, verbose=False)
        deltas, state_ptrs = get_leaf_deltas(tHMMobj) # gets the deltas matrix
        get_nonleaf_deltas(tHMMobj, deltas, state_ptrs)
        all_states = Viterbi(tHMMobj, deltas, state_ptrs)        

        acc_h3 = []
        cell_h3 = []
        bern_h3 = []
        bern_MAS_h3 = []
        bern_2_h3 = []
        cGom_MAS_h3 = []
        cGom_2_h3 = []
        scaleGom_MAS_h3 = []
        scaleGom_2_h3 = []
        EL_MAS_h3 = []
        EL_2_h3 = []
        KL_h3 = []
        
        for lin in range(tHMMobj.numLineages):
            lineage = tHMMobj.population[lin]
            T = tHMMobj.paramlist[lin]["T"]
            E = tHMMobj.paramlist[lin]["E"]
            EL = tHMMobj.EL[lin]

            
            #assign state 1 and state 2
            T_non_diag = np.zeros(numStates)
            for state_j in range(numStates):
                for state_k in range(numStates):
                    if state_j != state_k:
                        T_non_diag[state_j] = T[state_j,state_k]
            
            state_1 = np.argmin(T_non_diag) #state_MAS
            state_2 = np.argmax(T_non_diag)
            
            
            wrong = 0
            for cell in range(len(lineage)):
                if cell < len(masterLineage):
                    if all_states[lin][cell] == state_1:
                        pass
                    else:
                        wrong += 1
                elif cell >= len(masterLineage) and cell < len(newLineage):
                    if all_states[lin][cell] == state_2:
                        pass
                    else:
                        wrong += 1           
            
            accuracy = (len(lineage) - wrong)/len(lineage) #must be fixed for more than 1 lineage

            
            acc_h3.append(accuracy)
            cell_h3.append(len(lineage))
            print('h3',cell_h3)
            bern_MAS_h3.append(E[state_1,0])
            print('M',E[state_1,0])
            bern_2_h3.append(E[state_2,0])
            print('2',E[state_2,0])
            cGom_MAS_h3.append(E[state_1,1])
            cGom_2_h3.append(E[state_2,1])
            scaleGom_MAS_h3.append(E[state_1,2])
            scaleGom_2_h3.append(E[state_2,2])
            EL_MAS_h3.append(EL[:,state_1])
            EL_2_h3.append(EL[:,state_2])
            KL_h3.append(entropy(EL[:,state_1],EL[:,state_2]))
        
        acc_h2.extend(acc_h3)
        cell_h2.extend(cell_h3)
        print('h2', cell_h2)
        bern_MAS_h2.extend(bern_MAS_h3)
        bern_2_h2.extend(bern_2_h3)
        cGom_MAS_h2.extend(cGom_MAS_h3)
        cGom_2_h2.extend(cGom_2_h3)
        scaleGom_MAS_h2.extend(scaleGom_MAS_h3)
        scaleGom_2_h2.extend(scaleGom_2_h3)
        EL_MAS_h2.extend(EL_MAS_h3)
        EL_2_h2.extend(EL_2_h3)
        print('KL_h3',KL_h3)
        KL_h2.extend(KL_h3)
        
    acc_h1.extend(acc_h2)
    cell_h1.extend(cell_h2)
    print('h1',cell_h1)
    bern_MAS_h1.extend(bern_MAS_h2)
    bern_2_h1.extend(bern_2_h2)
    cGom_MAS_h1.extend(cGom_MAS_h2)
    cGom_2_h1.extend(cGom_2_h2)
    scaleGom_MAS_h1.extend(scaleGom_MAS_h2)
    scaleGom_2_h1.extend(scaleGom_2_h2)
    EL_MAS_h1.extend(EL_MAS_h2)
    EL_2_h1.extend(EL_2_h2)
    KL_h1.extend(KL_h2)

    
#for population in range(len(EL_MAS_h1)):
#KL.append(entropy(EL_MAS_h1[population],EL_2_h1[population]))

x=KL_h1
print(max(x))
fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
ax = axs[0,0]
ax.errorbar(x, acc_h1, fmt='o', c='b',marker="*",fillstyle='none')
ax.axhline(y=1, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='b',)
ax.set_title('Accuracy')
#ax.locator_params(nbins=4)

ax = axs[0,1]
ax.errorbar(x, bern_MAS_h1, fmt='o', c='b',marker="^",fillstyle='none')
ax.errorbar(x, bern_2_h1, fmt='o', c='g',marker="^",fillstyle='none')
ax.set_title('Bernoulli Parameter')
ax.axhline(y= MASlocBern, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='b')
ax.axhline(y=locBern2, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='g')

ax = axs[1,0]
ax.set_xlabel('KL Divergence')
ax.errorbar(x,cGom_MAS_h1, fmt='o',c='b',marker=(8,2,0))
ax.errorbar(x,cGom_2_h1, fmt='o',c='g',marker=(8,2,0))
ax.axhline(y=MAScGom, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='b')
ax.axhline(y=cGom2, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='g')
ax.set_title('Gompertz C')


ax = axs[1,1]
ax.set_xlabel('KL Divergence')
ax.errorbar(x,scaleGom_MAS_h1, fmt='o',c='b', marker="1")
ax.errorbar(x,scaleGom_2_h1, fmt='o',c='g', marker="1")
ax.axhline(y=MASscaleGom, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='b')
ax.axhline(y=scaleGom2, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='g')
ax.set_title('Gompertz Scale')

fig.suptitle('KL Divergence between intralineage cell types')

plt.savefig('TEST_KLdiv_classification.png')