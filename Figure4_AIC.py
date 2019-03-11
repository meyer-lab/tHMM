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

from Lin_adam import Lin_adam

from Analyze import Analyze
from Accuracy import Accuracy
from Matplot_gen import Matplot_gen

T_MAS = 130
T_2 = 61
switchT = 130
experimentTime = 130+30
lineage_num = [1]

state1 = 1
state2 = 4
states = range(state1,state2+1) 
reps = 1

MASinitCells = [1]
MASlocBern = [0.8]
MAScGom = [1.6]
MASscaleGom = [40]
initCells2 = [1]
locBern2 = [0.99]
cGom2 = [1.6]
scaleGom2 = [18]

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

X,lin2 = Lin_adam(experimentTime, lineage_num, MASlocBern, MAScGom, MASscaleGom, switchT, locBern2, cGom2, scaleGom2) 
while len(X) > 1000 or len(X) < 600 or (len(X)-len(lin2)) < 100 or (len(lin2)) < 100:
        X,lin2 = Lin_adam(experimentTime, lineage_num, MASlocBern, MAScGom, MASscaleGom, switchT, locBern2, cGom2, scaleGom2)

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
        deltas, state_ptrs, all_states, tHMMobj, NF, LL = Analyze(X, numStates)

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
ax.set_ylabel('Cost function',rotation=90)

#ax.locator_params(nbins=4)

fig.suptitle('Akaike Information Criterion')

plt.savefig('TEST_AIC_classification.png')