import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.stats import entropy


from lineage.BaumWelch import fit
from lineage.DownwardRecursion import get_root_gammas, get_nonroot_gammas
from lineage.Viterbi import get_leaf_deltas, get_nonleaf_deltas, Viterbi
from lineage.UpwardRecursion import get_leaf_Normalizing_Factors, get_leaf_betas, get_nonleaf_NF_and_betas, calculate_log_likelihood
from lineage.tHMM import tHMM
from lineage.tHMM_utils import max_gen, get_gen, get_parents_for_level
from lineage.Lineage_utils import remove_NaNs, get_numLineages, init_Population
from lineage.Lineage_utils import generatePopulationWithTime as gpt
from lineage.CellNode import CellNode



T_MAS = 60
T_2 = 60

state1 = 1
state2 = 4
states = range(state1,state2+1) 
reps = 1

'''MASinitCells = [1]
MASlocBern = [0.7]
MAScGom = [0.5]
MASscaleGom = [30]
initCells2 = [1]
locBern2 = [0.7]
cGom2 = [.5]
scaleGom2 = [30]'''

MASinitCells = [1]
MASlocBern = [0.99999999999]
MAScGom = [2]
MASscaleGom = [30]
initCells2 = [1]
locBern2 = [0.7]
cGom2 = [1.5]
scaleGom2 = [25]

acc_h1 = [] #list of lists of lists
cell_h1 = []
bern_MAS_h1 = []
bern_2_h1 = []
cGom_MAS_h1 = []
cGom_2_h1 = []
scaleGom_MAS_h1 = []
scaleGom_2_h1 = []
#LL_h1 = []
AIC_h1 = []


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
entropy = entropy()

for numStates in states: #a pop with num number of lineages
    print('numstates', numStates)
    acc_h2 = []
    cell_h2 = []
    bern_h2 = []
    bern_MAS_h2 = []
    bern_2_h2 = []
    cGom_MAS_h2 = []
    cGom_2_h2 = []
    scaleGom_MAS_h2 = []
    scaleGom_2_h2 = []
    LL_h2 = []
    
    for rep in range(reps):
        numStates = numStates
        tHMMobj = tHMM(X, numStates=numStates) # build the tHMM class with X
        fit(tHMMobj, max_iter=200, verbose=False)
        deltas, state_ptrs = get_leaf_deltas(tHMMobj) # gets the deltas matrix
        get_nonleaf_deltas(tHMMobj, deltas, state_ptrs)
        all_states = Viterbi(tHMMobj, deltas, state_ptrs)
        
        NF = get_leaf_Normalizing_Factors(tHMMobj)
        betas = get_leaf_betas(tHMMobj, NF)
        get_nonleaf_NF_and_betas(tHMMobj, NF, betas)
        LL = calculate_log_likelihood(tHMMobj, NF)

        acc_h3 = []
        cell_h3 = []
        bern_h3 = []
        bern_MAS_h3 = []
        bern_2_h3 = []
        cGom_MAS_h3 = []
        cGom_2_h3 = []
        scaleGom_MAS_h3 = []
        scaleGom_2_h3 = []
        LL_h3 = []
        
        for lin in range(tHMMobj.numLineages):
            lineage = tHMMobj.population[lin]
            T = tHMMobj.paramlist[lin]["T"]
            E = tHMMobj.paramlist[lin]["E"]

            
            cell_h3.append(len(lineage))
            print('h3',cell_h3)
            LL_h3.append(LL[lin])
        
        acc_h2.extend(acc_h3)
        cell_h2.extend(cell_h3)
        print('h2', cell_h2)
        LL_h2.extend(LL_h3)
        
    acc_h1.extend(acc_h2)
    cell_h1.extend(cell_h2)
    print('h1',cell_h1)
    
    AIC_h2 = []
    for rep_num in range(len(LL_h2)):
        AIC = -2*LL_h2[rep_num] + 2*(numStates*(numStates-1))
        AIC_h2.append(AIC)
    print('aic_h2',AIC_h2, 'LL', LL)
    AIC_h1.extend(AIC_h2)
    
    #LL_h1.append(np.sum(LL_h2))

    
#for population in range(len(EL_MAS_h1)):
#KL.append(entropy(EL_MAS_h1[population],EL_2_h1[population]))

x=states
print(max(x))
fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True)
ax = axs
ax.errorbar(x, AIC_h1, fmt='o', c='b',marker="*",fillstyle='none')
#ax.axhline(y=1, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='b',)
ax.set_title('AIC')
ax.set_xlabel('Number of States')
#ax.locator_params(nbins=4)

fig.suptitle('Akaike Information Criterion')

plt.savefig('TEST_AIC_classification.png')