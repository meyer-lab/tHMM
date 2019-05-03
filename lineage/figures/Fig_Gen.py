'''Generates 4 types of figures: Lineage Length, Number of Lineages in a Popuation, KL Divergence effects, and AIC Calculation. Currently, only the Depth_Two_State_Lineage is used.'''

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from .Depth_Two_State_Lineage import Depth_Two_State_Lineage
from .Breadth_Two_State_Lineage import Breadth_Two_State_Lineage
from ..Analyze import Analyze
from .Matplot_gen import Matplot_gen
from ..tHMM_utils import getAccuracy, getAIC
from ..Lineage_utils import remove_singleton_lineages


def Lineage_Length(T_MAS=975, T_2=175, reps=2, MASinitCells=[1], MASlocBern=[0.99], MASbeta=[200], initCells2=[1],
                   locBern2=[0.8], beta2=[25], numStates=2, max_lin_length=500, min_lin_length=200, FOM='E', verbose=False):
    '''This has been modified for an exonential distribution'''
    '''Creates four figures of how accuracy, bernoulli parameter, gomp c, and gomp scale change as the number of cells in a single lineage is varied'''

    acc_h1 = []  # list of lists of lists
    cell_h1 = []
    bern_MAS_h1 = []
    bern_2_h1 = []
    cGom_MAS_h1 = []
    cGom_2_h1 = []
    scaleGom_MAS_h1 = []
    scaleGom_2_h1 = []

    for rep in range(reps):
        print('Rep:', rep)
        print(FOM)
        X = Breadth_Two_State_Lineage(experimentTime=T_MAS+T_2, initCells=MASinitCells, locBern=MASlocBern, betaExp=MASbeta, switchT=T_2, bern2=locBern2, betaExp2=beta2, FOM=FOM, verbose=False)

        while len(X) > max_lin_length or len(X) < min_lin_length:
            X = Breadth_Two_State_Lineage(experimentTime=T_MAS+T_2, initCells=MASinitCells, locBern=MASlocBern, betaExp=MASbeta, switchT=T_2, bern2=locBern2, betaExp2=beta2, FOM=FOM, verbose=False)
            '''MUST be made a function'''

        print('X', len(X))
        _, _, all_states, tHMMobj, _, _ = Analyze(X, numStates)
        print('analyzed')
        acc_h2 = []
        cell_h2 = []
        bern_MAS_h2 = []
        bern_2_h2 = []
        cGom_MAS_h2 = []
        cGom_2_h2 = []
        scaleGom_MAS_h2 = []
        scaleGom_2_h2 = []
        for lin in range(tHMMobj.numLineages):
            AccuracyPop, _, stateAssignmentPop = getAccuracy(tHMMobj, all_states, verbose=False)
            accuracy = AccuracyPop[lin]
            print('accuracy:{}'.format(accuracy))
            state_1 = stateAssignmentPop[lin][0]
            state_2 = stateAssignmentPop[lin][1]
            lineage = tHMMobj.population[lin]
            T = tHMMobj.paramlist[lin]["T"]
            E = tHMMobj.paramlist[lin]["E"]
            pi = tHMMobj.paramlist[lin]["pi"]

            acc_h2.append(accuracy * 100)
            cell_h2.append(len(lineage))
            bern_MAS_h2.append(E[state_1, 0])
            bern_2_h2.append(E[state_2, 0])
            cGom_MAS_h2.append(E[state_1, 1])
            cGom_2_h2.append(E[state_2, 1])
            if FOM == 'G':
                scaleGom_MAS_h2.append(E[state_1, 2])
                scaleGom_2_h2.append(E[state_2, 2])

            if verbose:
                print('pi', pi)
                print('T', T)
                print('E', E)
                print('accuracy:', accuracy)
                print('MAS length, 2nd lin length:', len(masterLineage), len(newLineage) - len(masterLineage))

        acc_h1.extend(acc_h2)
        cell_h1.extend(cell_h2)
        bern_MAS_h1.extend(bern_MAS_h2)
        bern_2_h1.extend(bern_2_h2)
        cGom_MAS_h1.extend(cGom_MAS_h2)
        cGom_2_h1.extend(cGom_2_h2)
        scaleGom_MAS_h1.extend(scaleGom_MAS_h2)
        scaleGom_2_h1.extend(scaleGom_2_h2)

    x = cell_h1
    if FOM == 'G':
        data = (x, acc_h1, bern_MAS_h1, bern_2_h1, MASlocBern, locBern2, cGom_MAS_h1, cGom_2_h1, MAScGom, cGom2, scaleGom_MAS_h1, scaleGom_2_h1, MASscaleGom, scaleGom2)
    elif FOM == 'E':
        data = (x, acc_h1, bern_MAS_h1, bern_2_h1, MASlocBern, locBern2, cGom_MAS_h1, cGom_2_h1, MASbeta, beta2, scaleGom_MAS_h1, scaleGom_2_h1)

    return data


def Lineages_per_Population_Figure(lineage_start=1, lineage_end=4, numStates=2, T_MAS=130, T_2=61, reps=1, MASinitCells=[1], MASlocBern=[0.8], MASbetaExp=[
                                   40], initCells2=[1], locBern2=[0.99], betaExp2=[18], max_lin_length=300, min_lin_length=50, verbose=True):
    '''Creates four figures of how accuracy, bernoulli parameter, gomp c, and gomp scale change as the number of lineages in a population are varied'''
    if verbose:
        print('starting')
    lineages = range(lineage_start, lineage_end + 1)
    acc_h1 = []  # list of lists of lists
    cell_h1 = []
    bern_MAS_h1 = []
    bern_2_h1 = []
    betaExp_MAS_h1 = []
    betaExp_2_h1 = []
    lineage_h1 = []

    for lineage_num in lineages:  # a pop with num number of lineages
        acc_h2 = []
        cell_h2 = []
        bern_MAS_h2 = []
        bern_2_h2 = []
        betaExp_MAS_h2 = []
        betaExp_2_h2 = []

        for rep in range(reps):
            X1 = []
            if verbose:
                print('making lineage')

            for num in range(lineage_num):
                X, masterLineage, newLineage = Depth_Two_State_Lineage(T_MAS, MASinitCells, MASlocBern, MASbetaExp, T_2, initCells2, locBern2, betaExp2)
                while len(newLineage) > max_lin_length or len(masterLineage) < min_lin_length or (len(newLineage) - len(masterLineage)) < min_lin_length:
                    X, masterLineage, newLineage = Depth_Two_State_Lineage(T_MAS, MASinitCells, MASlocBern, MASbetaExp, T_2, initCells2, locBern2, betaExp2)
                X1.extend(newLineage)

            X = remove_singleton_lineages(X1)  # this is one single list with a number of lineages equal to what is inputted
            print(len(X))
            _, _, all_states, tHMMobj, _, _ = Analyze(X, numStates)
            acc_h3 = []
            cell_h3 = []
            bern_MAS_h3 = []
            bern_2_h3 = []
            betaExp_MAS_h3 = []
            betaExp_2_h3 = []

            if verbose:
                print('analyzing')
            for lin in range(tHMMobj.numLineages):
                AccuracyPop, _, stateAssignmentPop = getAccuracy(tHMMobj, all_states, verbose=False)
                accuracy = AccuracyPop[lin]
                state_1 = stateAssignmentPop[lin][0]
                state_2 = stateAssignmentPop[lin][1]
                lineage = tHMMobj.population[lin]
                T = tHMMobj.paramlist[lin]["T"]
                E = tHMMobj.paramlist[lin]["E"]
                pi = tHMMobj.paramlist[lin]["pi"]

                acc_h3.append(100 * accuracy)
                cell_h3.append(len(lineage))
                bern_MAS_h3.append(E[state_1, 0])
                bern_2_h3.append(E[state_2, 0])
                betaExp_MAS_h3.append(E[state_1, 1])
                betaExp_2_h3.append(E[state_2, 1])

                if verbose:
                    print('pi', pi)
                    print('T', T)
                    print('E', E)
                    print('accuracy:', accuracy)
                    print('MAS length, 2nd lin length:', len(masterLineage), len(newLineage) - len(masterLineage))

            acc_h2.extend(acc_h3)
            cell_h2.extend(cell_h3)
            bern_MAS_h2.extend(bern_MAS_h3)
            bern_2_h2.extend(bern_2_h3)
            betaExp_MAS_h2.extend(betaExp_MAS_h3)
            betaExp_2_h2.extend(betaExp_2_h3)

        acc_h1.append(np.mean(acc_h2))
        cell_h1.extend(cell_h2)
        bern_MAS_h1.append(np.mean(bern_MAS_h2))
        bern_2_h1.append(np.mean(bern_2_h2))
        betaExp_MAS_h1.append(np.mean(betaExp_MAS_h2))
        betaExp_2_h1.append(np.mean(betaExp_2_h2))
        lineage_h1.append(lineage_num)

        if verbose:
            print('Accuracy of', lineage_num, 'is', np.mean(acc_h2))

    x = lineage_h1
    data = (x, acc_h1, bern_MAS_h1, bern_2_h1, MASlocBern, locBern2, betaExp_MAS_h1, betaExp_2_h1, MASbetaExp, betaExp2)
    return data


def AIC_Figure(T_MAS=130, T_2=61, state1=1, state2=4, reps=1, MASinitCells=[1], MASlocBern=[0.8], MASbetaExp=[40],
               initCells2=[1], locBern2=[0.99], betaExp2=[18], max_lin_length=1500, min_lin_length=100, verbose=False):
    '''Calculates and plots an AIC for all inputted states'''

    states = range(state1, state2 + 1)
    acc_h1 = []  # list of lists of lists
    cell_h1 = []
    AIC_h1 = []
    X, masterLineage, newLineage = Depth_Two_State_Lineage(T_MAS, MASinitCells, MASlocBern, MASbetaExp, T_2, initCells2, locBern2, betaExp2)

    while len(newLineage) > max_lin_length or len(masterLineage) < min_lin_length or (len(newLineage) - len(masterLineage)) < min_lin_length:
        X, masterLineage, newLineage = Depth_Two_State_Lineage(T_MAS, MASinitCells, MASlocBern, MASbetaExp, T_2, initCells2, locBern2, betaExp2)

    if verbose:
        print(len(masterLineage), len(newLineage))

    for numStates in states:  # a pop with num number of lineages
        _, _, all_states, tHMMobj, _, LL = Analyze(X, numStates)
        AccuracyPop, _, _ = getAccuracy(tHMMobj, all_states, verbose=False)
        acc_h2 = []
        cell_h2 = []
        AIC_h2 = []

        for rep in range(reps):
            AIC_value_holder_rel_0 = getAIC(tHMMobj, LL)
            acc_h3 = []
            cell_h3 = []
            AIC_h3 = []

            for lin in range(tHMMobj.numLineages):
                cell_h3.append(len(tHMMobj.population[lin]))
                acc_h3.append(100 * AccuracyPop[lin])
                AIC_h3.append(AIC_value_holder_rel_0[lin])

            cell_h2.extend(cell_h3)
            acc_h2.append(acc_h3)
            AIC_h2.append(AIC_value_holder_rel_0)

        acc_h1.extend(acc_h2)
        cell_h1.extend(cell_h2)
        AIC_h1.extend(AIC_h2)

        if verbose:
            print('h1', cell_h1)

    x = states
    fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True)
    ax = axs
    ax.errorbar(x, AIC_h1, fmt='o', c='b', marker="*", fillstyle='none')
    ax.set_title('AIC')
    ax.set_xlabel('Number of States')
    ax.set_ylabel('Cost function', rotation=90)
    fig.suptitle('Akaike Information Criterion')
    plt.savefig('TEST_AIC_classification.png')
