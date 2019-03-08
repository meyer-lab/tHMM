import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

from Lin_shak import Lin_shak
from Analyze import Analyze
from Accuracy import Accuracy
from Matplot_gen import Matplot_gen

################ Number of cell in a single lineage

T_MAS = 100
T_2 = 81
times = range(1,2) 
reps = 20
switchT = 25

MASinitCells = [1]
MASlocBern = [0.99999999999]
MAScGom = [2]
MASscaleGom = [30]
initCells2 = [1]
locBern2 = [0.8] #0.7
cGom2 = [1.8] #1.5
scaleGom2 = [27] #25

numStates = 2

acc_h1 = [] #list of lists of lists
cell_h1 = []
bern_MAS_h1 = []
bern_2_h1 = []
cGom_MAS_h1 = []
cGom_2_h1 = []
scaleGom_MAS_h1 = []
scaleGom_2_h1 = []

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
    
    for rep in range(reps):
        print('Rep:', rep)       
        
        X, masterLineage, newLineage = Lin_shak(T_MAS, MASinitCells, MASlocBern, MAScGom, MASscaleGom, T_2, initCells2, locBern2, cGom2, scaleGom2) 
        while len(newLineage) > 2000:
            X, masterLineage, newLineage = Lin_shak(T_MAS, MASinitCells, MASlocBern, MAScGom, MASscaleGom, T_2, initCells2, locBern2, cGom2, scaleGom2)
            
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
        
        for lin in range(tHMMobj.numLineages):
                     
            T,E,pi,state_1,state_2,accuracy,lineage = Accuracy(tHMMobj, lin, numStates, masterLineage, newLineage, all_states)
            
            acc_h3.append(accuracy*100)
            print('pi',pi)
            print('T',T)
            print('accuracy:',accuracy)
            print('MAS, 2nd lin:',len(masterLineage),len(newLineage)-len(masterLineage))
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
        
        acc_h2.extend(acc_h3)
        cell_h2.extend(cell_h3)
        print('h2', cell_h2)
        bern_MAS_h2.extend(bern_MAS_h3)
        bern_2_h2.extend(bern_2_h3)
        cGom_MAS_h2.extend(cGom_MAS_h3)
        cGom_2_h2.extend(cGom_2_h3)
        scaleGom_MAS_h2.extend(scaleGom_MAS_h3)
        scaleGom_2_h2.extend(scaleGom_2_h3)
        
    acc_h1.extend(acc_h2)
    cell_h1.extend(cell_h2)
    print('h1',cell_h1)
    bern_MAS_h1.extend(bern_MAS_h2)
    bern_2_h1.extend(bern_2_h2)
    cGom_MAS_h1.extend(cGom_MAS_h2)
    cGom_2_h1.extend(cGom_2_h2)
    scaleGom_MAS_h1.extend(scaleGom_MAS_h2)
    scaleGom_2_h1.extend(scaleGom_2_h2)

x=cell_h1
print(acc_h1)
Matplot_gen(x,acc_h1,bern_MAS_h1,bern_2_h1,MASlocBern,locBern2,cGom_MAS_h1,cGom_2_h1,MAScGom,           cGom2,scaleGom_MAS_h1,scaleGom_2_h1,MASscaleGom,scaleGom2, xlabel = 'Number of Cells', title = 'Cells in a Lineage', save_name = 'Figure1.png')