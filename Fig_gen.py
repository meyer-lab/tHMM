import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

from Lin_shak import Lin_shak
from Analyze import Analyze
from Matplot_gen import Matplot_gen

from .lineage.tHMM_utils.py import getAccuracy

'''Generates 4 types of figures: Lineage Length, Number of Lineages in a Popuation, KL Divergence effects, and AIC Calculation'''

def Lineage_Length(T_MAS = 130, T_2 = 61, reps = 20, switchT = 25, MASinitCells = [1], MASlocBern = [0.8], MAScGom = [1.6], MASscaleGom = [40], initCells2 = [1], locBern2 = [0.99], cGom2 = [1.6], scaleGom2 = [18], numStates = 2, max_lin_length, min_lin_length):

    '''Creates four figures of how accuracy, bernoulli parameter, gomp c, and gomp scale change as the number of cells in a single lineage is varied'''

    acc_h1 = [] #list of lists of lists
    cell_h1 = []
    bern_MAS_h1 = []
    bern_2_h1 = []
    cGom_MAS_h1 = []
    cGom_2_h1 = []
    scaleGom_MAS_h1 = []
    scaleGom_2_h1 = []

    for rep in range(reps):
        print('Rep:', rep)       
        X, masterLineage, newLineage = Lin_shak(T_MAS, MASinitCells, MASlocBern, MAScGom, MASscaleGom, T_2, initCells2, locBern2, cGom2, scaleGom2) 
        while len(newLineage) > max_lin_length or len(masterLineage) < min_lin_length or (len(newLineage)-len(masterLineage)) < min_lin_length:
            X, masterLineage, newLineage = Lin_shak(T_MAS, MASinitCells, MASlocBern, MAScGom, MASscaleGom, T_2, initCells2, locBern2, cGom2, scaleGom2)
        deltas, state_ptrs, all_states, tHMMobj, NF, LL = Analyze(X, numStates)
        acc_h2 = []
        cell_h2 = []
        bern_h2 = []
        bern_MAS_h2 = []
        bern_2_h2 = []
        cGom_MAS_h2 = []
        cGom_2_h2 = []
        scaleGom_MAS_h2 = []
        scaleGom_2_h2 = []         
        for lin in range(tHMMobj.numLineages):
            getAccuracy(tHMMobj, all_states, verbose=False)
            accuracy = tHMM.Accuracy[lin]
            lineage = tHMMobj.population[lin]
            T = tHMMobj.paramlist[lin]["T"]
            E = tHMMobj.paramlist[lin]["E"]
            pi = tHMMobj.paramlist[lin]["pi"]

            acc_h2.append(accuracy*100)
            cell_h2.append(len(lineage))
            bern_MAS_h2.append(E[state_1,0])
            bern_2_h2.append(E[state_2,0])
            cGom_MAS_h2.append(E[state_1,1])
            cGom_2_h2.append(E[state_2,1])
            scaleGom_MAS_h2.append(E[state_1,2])
            scaleGom_2_h2.append(E[state_2,2])
            print('pi',pi)
            print('T',T)
            print('E',E)
            print('accuracy:',accuracy)
            print('MAS length, 2nd lin length:',len(masterLineage),len(newLineage)-len(masterLineage))
        acc_h1.extend(acc_h2)
        cell_h1.extend(cell_h2)
        bern_MAS_h1.extend(bern_MAS_h2)
        bern_2_h1.extend(bern_2_h2)
        cGom_MAS_h1.extend(cGom_MAS_h2)
        cGom_2_h1.extend(cGom_2_h2)
        scaleGom_MAS_h1.extend(scaleGom_MAS_h2)
        scaleGom_2_h1.extend(scaleGom_2_h2)
    x=cell_h1
    print(acc_h1)
    Matplot_gen(x,acc_h1,bern_MAS_h1,bern_2_h1,MASlocBern,locBern2,cGom_MAS_h1,cGom_2_h1,MAScGom,           cGom2,scaleGom_MAS_h1,scaleGom_2_h1,MASscaleGom,scaleGom2, xlabel = 'Number of Cells', title = 'Cells in a Lineage', save_name = 'Lineage_Length_Figure.png')
    data = np.array([x,acc_h1,bern_MAS_h1,bern_2_h1,MASlocBern,locBern2,cGom_MAS_h1,cGom_2_h1,MAScGom,           cGom2,scaleGom_MAS_h1,scaleGom_2_h1,MASscaleGom,scaleGom2])
    
    return(data)

def Lineages_per_Population_Figure():
    '''Creates four figures of how accuracy, bernoulli parameter, gomp c, and gomp scale change as the number of lineages in a population are varied'''
    