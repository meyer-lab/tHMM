""" The file contains the methods used to input lineage data from the Heiser lab. """

import pandas as pd
import math
from .CellVar import CellVar as c


def import_Heiser(path=r"lineage/data/heiser_data/LT_AU003_A3_4_Lapatinib_V2.xlsx"):
    """
    Imports data from the Heiser lab
    Outputs a list of lists containing cells containing observations from the Excel file
    In this particular dataset, we look at the following phenotypes:
    
    1. survival past G1 (death or transition into G2),
    2. survival past G2 (death or division into daughter cells),
    3. time spent in G1,
    4. time spent in G2,
    5. time spent totally alive
    
    The observations for each cell are stored in cell as the following list in the observation (obs) attribute:
    
    1. Boolean (survived G1, None if G1 didn't happen), 
    2. Boolean (survived G2, None if G2 didn't happen), 
    3. Double (time spent in G1), 
    4. Double (time spent in G2),
    5. Double (continuous time at cell division Nan/145 if division didn't occur) this helps with internal calculations]
    """
    excel_file = pd.read_excel(path, header=None)
    data = excel_file.to_numpy()

    # position of Lineage Size attribute
    lineageSizeIndex = 0
    # current Lineage Posistion
    lPos = 0
    # current Lineage Number
    lineageNo = 0
    # Next Upper range value
    nextUp = 1
    # find Lineage Size attribute
    while data[0][lineageSizeIndex] != "Lineage Size":
        lineageSizeIndex += 1

    # find Lineages
    lineages = []
    while lPos < len(data):
        # increment to find next lineage (so it doesn't find the same one)
        lPos += 1
        # find Next Lineage Position
        while(lPos < len(data) and math.isnan(data[lPos][0])):
            lPos += 1

        #checking if file for errors (if lineage# lines up)
        if(lPos<len(data)):
            lineageNo += 1
            assert lineageNo == data[lPos][0]

        # determine if lineage has cells
        if(lPos < len(data) and not math.isnan(data[lPos][1])):
            # add list for the lineage
            currentLineage = []
            # make Parent
            parentCell = c(parent=None, gen=1, synthetic=False)
            divisionTime = data[lPos][1+2]
            parentCell.obs = [0,0,0,0,0,0] 

            #[x  x] case
            if data[lPos][1] == data[lPos][1+2]:

                #Time Censored [145  145]
                if (data[lPos][1] == 145):  
                    parentCell.obs[0] = float('nan')  #live/die G1
                    parentCell.obs[4] = 0
                #Not Time Censored [x=/=145   x=/=145]
                else:
                    parentCell.obs[0] = 0
                    parentCell[4] = 1
                    
                parentCell.obs[1] = float('nan') #Did not go to G2
                parentCell.obs[2] = data[lPos][1] #Time Spent in G1
                parentCell.obs[3] = float('nan') #Spent no time in G2
                parentCell.obs[5] = float('nan') #G2 outcome unknown
            
            #[x  y]/[x y  ] case (general)
            else:
                #[1  y]/[1 y  ] case
                if data[lPos][1] == 1:
                    parentCell.obs[0]  = 1  #did not start in G1
                    parentCell.obs[2] = float('nan') #Spent no time in G1
                    parentCell.obs[4] = float('nan') #G1 outcome unknown
                    
                #[x=/=1   y]/[x=/=1 y  ] case
                else:
                    parentCell.obs[0] = 1  #survived G1
                    parentCell.obs[2] = data[lPos][1] #Time spent in G1
                    parentCell.obs[4] = 1 #G1 uncensored

                #[x  y] case (general)
                if math.isnan(data[lPos][1 + 1]):
                    parentCell.obs[1] = float('nan') if (data[lPos][1+2] == 145) else 1  #survived G2
                    parentCell.obs[3] = data[lPos][1+2] if (math.isnan(parentCell.obs[2])) else data[lPos][1+2]-parentCell.obs[2] #Time spent in G2
                
                #[x y  ] case
                else:
                    parentCell.obs[1] = 0 #died in G2
                    parentCell.obs[3] = data[lPos][1+1]-parentCell.obs[2] #Time spent in G2

                #Time Censored Case  [x  145]
                parentCell.obs[5] = 0 if (data[lPos][1+2] == 145 or data[lPos][1] == 1) else 1 #Censored if the cell started in G2 or total time is 145

            # find lower value of range and store next upper
            upper = nextUp
            nextUp += 1
            while(nextUp < len(data) and math.isnan(data[nextUp][lineageSizeIndex])):
                nextUp += 1
            if nextUp == len(data):
                lower = nextUp - 1
            else:
                lower = nextUp - 2
            # find upper daughter and recurse
            parentCell.left = tryRecursion(1, lPos, upper, parentCell, currentLineage, lineageSizeIndex, data,divisionTime, True)
            # find lower daughter and recurse
            parentCell.right = tryRecursion(1, lower, lPos, parentCell, currentLineage, lineageSizeIndex, data, divisionTime, False)

            # add first generation to lineage (apparently python passes by reference for objects so this probably can be done before or after)
            currentLineage.append(parentCell)

            # store lineage in list of lineages
            lineages.append(currentLineage)
    return lineages



def tryRecursion(pColumn, lower, upper, parentCell, currentLineage, lineageSizeIndex, data, divisionTime, firstHalf):
    """
    Method for Top and Bottom halves of the Lineage Tree as recorded in the Excel files
    (They mirrored the posistions for the last set of daughter cells...)
    In the excel files given the top and bottom halves of the tree are mirrored
    This means that using the same method for import of the whole tree will not work correctly when the tree is full
    the ranges the recursionB method searches in have to be offset by 1 
    so that the algorithm will search the proper positions for the last possible generation of cells
    """
    found = False
    # check if this is the last possible cell
    if pColumn + 3 >= lineageSizeIndex:
        return None
    # find branch within provided range
    
    pColumn += 3

    #this will properly offset the range based on whether the algorithm is searching the top half or bottom half of the tree
    if firstHalf:
        u = upper
        l = lower
        
    else:
        u = upper+1
        l = lower+1

    for parentPos in range(u, l):
        if not math.isnan(data[parentPos][pColumn]):
            found = True
            break
    if not found:
        return None
    # store values into lineage here
    daughterCell = c(parent=parentCell, gen=parentCell.gen + 1, synthetic=parentCell.synthetic)
    daughterCell.obs = [ 0, 0, 0, 0, 0, 0 ] # This stores the Time at cell division

    #[x  x] case
    if data[parentPos][pColumn] == data[parentPos][pColumn+2]:
        
        # Time Censored [145  145]
        if (data[pColumn][pColumn] == 145):
            daughterCell.obs[0] = float('nan')   #We don't know the outcome of G1
            daughterCell.obs[4] = 0  #G1 censored
        
        #Not Time Censored [x=/=145   x=/=145]
        else:
            daughterCell.obs[0] = 0  #G1 death
            daughterCell.obs[4] = 1  #G1 uncensored

        daughterCell.obs[1] = float('nan')  #Did not go to G2
        daughterCell.obs[2] = data[parentPos][pColumn] - divisionTime#Time Spent in G1
        daughterCell.obs[3] = float('nan') #Spent no time in G2
        daughterCell.obs[5] = float('nan') #We don't have information about G2

    #[x  y]/[x y  ] case (general)
    else:
        #[1  y]/[1 y  ] case is not possible anymore
        daughterCell.obs[0] =  1  #survived G1
        daughterCell.obs[2] = data[parentPos][pColumn] -  divisionTime#Time spent in G1
        daughterCell.obs[4] = 1  #G1 uncensored

        #[x  y] case (general)
        if math.isnan(data[parentPos][pColumn + 1]):
            daughterCell.obs[1] = float('nan') if (data[parentPos][pColumn+2] == 145) else 1  #survived G2
            daughterCell.obs[3] = data[parentPos][pColumn+2]-data[parentPos][pColumn] #Time spent in G2
        #[x y  ] case
        else:
            daughterCell.obs[1] = 0 #died in G2
            daughterCell.obs[3] = data[parentPos][pColumn+1]-data[parentPos][pColumn] #Time spent in G2
        #Time Censored Case ([x  145])
        daughterCell.obs[5] = 0 if (data[parentPos][pColumn+2] == 145) else 1  #Censored if final time is 145, otherise uncensored

    # find upper daughter
    daughterCell.left = tryRecursion(pColumn, parentPos, upper, daughterCell, currentLineage, lineageSizeIndex, data, data[parentPos][pColumn+2], firstHalf)
    # find lower daughter
    daughterCell.right = tryRecursion(pColumn, lower, parentPos, daughterCell, currentLineage, lineageSizeIndex, data, data[parentPos][pColumn+2], firstHalf)

    # add daughter to current Lineage
    currentLineage.append(daughterCell)
    return daughterCell


