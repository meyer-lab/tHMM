import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.ticker


from lineage.BaumWelch import fit
from lineage.DownwardRecursion import get_root_gammas, get_nonroot_gammas
from lineage.Viterbi import get_leaf_deltas, get_nonleaf_deltas, Viterbi
from lineage.UpwardRecursion import get_leaf_Normalizing_Factors, get_leaf_betas, get_nonleaf_NF_and_betas
from lineage.tHMM import tHMM
from lineage.tHMM_utils import max_gen, get_gen, get_parents_for_level
from lineage.Lineage_utils import remove_NaNs, get_numLineages, init_Population
from lineage.Lineage_utils import generatePopulationWithTime as gpt
from lineage.CellNode import CellNode

def Matplot_gen(x,acc_h1,bern_MAS_h1,bern_2_h1,MASlocBern,locBern2,cGom_MAS_h1,cGom_2_h1,MAScGom,
                cGom2,scaleGom_MAS_h1,scaleGom_2_h1,MASscaleGom,scaleGom2, xlabel, title, save_name):
    
    print(max(x))
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
    ax = axs[0,0]
    ax.set_ylim(0,110)
    l1 = ax.errorbar(x, acc_h1, fmt='o', c='b',marker="*",fillstyle='none', label = 'Accuracy')
    ax.axhline(y=100, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='b',)
    ax.set_ylabel('Accuracy (%)',rotation=90)
    vals = ax.get_yticks()
    #ax.set_yticklabels([str(int(x)) + '%' for x in vals])

    ax = axs[0,1]
    l2 = ax.errorbar(x, bern_MAS_h1, fmt='o', c='r',marker="^",fillstyle='none', label = 'State 1')
    l3 = ax.errorbar(x, bern_2_h1, fmt='o', c='g',marker="^",fillstyle='none', label = 'State 2')
    ax.set_ylabel('Bernoulli', rotation=90)
    ax.axhline(y= MASlocBern, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='r')
    ax.axhline(y=locBern2, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='g')

    ax = axs[1,0]
    ax.set_xlabel(xlabel)
    ax.set_xscale("log", nonposx='clip')
    ax.errorbar(x,cGom_MAS_h1, fmt='o',c='r',marker="^",fillstyle='none', label = 'State 1')
    ax.errorbar(x,cGom_2_h1, fmt='o',c='g',marker="^",fillstyle='none', label = 'State 2')
    ax.axhline(y=MAScGom, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='r')
    ax.axhline(y=cGom2, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='g')
    ax.set_ylabel('Gompertz C',rotation=90)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())


    ax = axs[1,1]
    ax.set_xlabel(xlabel)
    ax.set_xscale("log", nonposx='clip')
    ax.errorbar(x,scaleGom_MAS_h1, fmt='o',c='r',marker="^",fillstyle='none', label = 'State 1')
    ax.errorbar(x,scaleGom_2_h1, fmt='o',c='g',marker="^",fillstyle='none', label = 'State 2')
    ax.axhline(y=MASscaleGom, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='r')
    ax.axhline(y=scaleGom2, linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth=1, color='g')
    ax.set_ylabel('Gompertz Scale',rotation=90)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(title)
    fig.savefig(save_name)



'''            state_1 = np.argmax(pi)
            state_2 = np.argmin(pi)
            
            trues = []
        
            
            wrong = 0
            for cell in range(len(lineage)):
                trues.append(lineage[cell].true_state)
                if lineage[cell].true_state == 0:
                    if all_states[lin][cell] == state_1:
                        pass
                    else:
                        wrong += 1
                elif lineage[cell].true_state == 1:
                    if all_states[lin][cell] == state_2:
                        pass
                    else:
                        wrong += 1           
            print('viterbi',all_states[lin])
            print('trues', trues)
            
            print('pi', pi)
            accuracy = (len(lineage) - wrong)/len(lineage) #must be fixed for more than 1 lineage
            
            
                    newLineage = gpt(75, MASinitCells, MASlocBern, MAScGom, MASscaleGom, switchT, locBern2, cGom2, scaleGom2)
        while len(newLineage) == 0:
            newLineage = gpt(experimentTime, MASinitCells, MASlocBern, MAScGom, MASscaleGom, switchT, locBern2, cGom2, scaleGom2)
        
        X = remove_NaNs(newLineage)
        print(len(newLineage))'''