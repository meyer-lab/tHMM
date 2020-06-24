""" The file contains the methods used to input lineage data from the Heiser lab. """

import pandas as pd
import math
from CellVar import CellVar as c, double


def import_Heiser(path=r"~/Projects/CAPSTONE/lineage/data/heiser_data/LT_AU003_A3_4_Lapatinib_V2.xlsx"):
    excel_file = pd.read_excel(path, header=None)
    data = excel_file.to_numpy()
    # Cell.obs stored as:
    #  [Boolean (survived G1, None if G1 didn't happen), 
    #   Boolean (survived G2, None if G2 didn't happen), 
    #   Double (time spent in G1), 
    #   Double (time spent in G2),
    #   Double (continuous time at cell division Nan/145 if division didn't occur) this helps with internal calculations]



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
        lineageNo += 1

        # determine if lineage has cells
        if(lPos < len(data) and not math.isnan(data[lPos][1])):
            # add list for the lineage
            currentLineage = []
            # make Parent
            parentCell = c(parent=None, gen=1, synthetic=False)

            parentCell.obs = [0,0,0,0,data[lPos][1+2]] #this stores the time of division 

            #[x  x] case
            if data[lPos][1] == data[lPos][1+2]:
                parentCell.obs[0] = (data[lPos][1] == 145) #live/die G1
                parentCell.obs[2] = data[lPos][1] #Time Spent in G1
                parentCell.obs[1] = None #Did not go to G2
                parentCell.obs[3] = 0 #Spent no time in G2
            
            #[x  y]/[x y  ] case (general)
            else:
                #[1  y]/[1 y  ] case
                if data[lPos][1] == 1:
                    parentCell.obs[0]  = None #did not start in G1
                    parentCell.obs[2] = 0 #Spent no time in G1
                    
                #[x=/=1   y]/[x=/=1 y  ] case
                else:
                    parentCell.obs[0] = True  #survived G1
                    parentCell.obs[2] = data[lPos][1] #Time spent in G1

                #[x  y] case (general)
                if math.isnan(data[lPos][1 + 1]):
                    parentCell.obs[1] = True  #survived G2
                    parentCell.obs[3] = data[lPos][1+2]-parentCell.obs[2] #Time spent in G2
                #[x y  ] case
                else:
                    parentCell.obs[1] = False #died in G2
                    parentCell.obs[3] = data[lPos][1+1]-parentCell.obs[2] #Time spent in G2

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
            parentCell.left = tryRecursionT(1, lPos, upper, parentCell, currentLineage, lineageSizeIndex, data)
            # find lower daughter and recurse
            parentCell.right = tryRecursionB(1, lower, lPos, parentCell, currentLineage, lineageSizeIndex, data)

            # add first generation to lineage (apparently python passes by reference for objects so this probably can be done before or after)
            currentLineage.append(parentCell)

            # store lineage in list of lineages
            lineages.append(currentLineage)
    return lineages


#In the excel files given the top and bottom halves of the tree are mirrored
#This means that using the same method for import of the whole tree will not work correctly when the tree is full
#the ranges the recursionB method searches in have to be offset by 1 
#so that the algorithm will search the proper positions for the last possible generation of cells
def tryRecursionT(pColumn, lower, upper, parentCell, currentLineage, lineageSizeIndex, data):
    """
    Method for Top half of Lineage Tree (They mirrored the posistions for the last set of daughter cells...)
    """
    found = False
    # check if this is the last possible cell
    if pColumn + 3 >= lineageSizeIndex:
        return None
    # find branch within provided range
    pColumn += 3
    for parentPos in range(upper, lower):
        if not math.isnan(data[parentPos][pColumn]):
            found = True
            break
    if not found:
        return None
    # store values into lineage here
    daughterCell = c(parent=parentCell, gen=parentCell.gen + 1, synthetic=parentCell.synthetic)
    daughterCell.obs = [ 0, 0, 0, 0, data[parentPos][pColumn+2] ] # This stores the Time at cell division

    #[x  x] case
    if data[parentPos][pColumn] == data[parentPos][pColumn+2]:
        daughterCell.obs[0] = (data[parentPos][pColumn] == 145) #live/die G1
        daughterCell.obs[2] = data[parentPos][pColumn] - parentCell.obs[4]#Time Spent in G1
        daughterCell.obs[1] = None #Did not go to G2
        daughterCell.obs[3] = 0 #Spent no time in G2

    #[x  y]/[x y  ] case (general)
    else:
        #[1  y]/[1 y  ] case is not possible anymore
        daughterCell.obs[0] = True  #survived G1
        daughterCell.obs[2] = data[parentPos][pColumn] -  parentCell.obs[4]#Time spent in G1

        #[x  y] case (general)
        if math.isnan(data[parentPos][pColumn + 1]):
            daughterCell.obs[1] = True  #survived G2
            daughterCell.obs[3] = data[parentPos][pColumn+2]-data[parentPos][pColumn] #Time spent in G2
        #[x y  ] case
        else:
            daughterCell.obs[1] = False #died in G2
            daughterCell.obs[3] = data[parentPos][pColumn+1]-data[parentPos][pColumn] #Time spent in G2
    # find upper daughter
    daughterCell.left = tryRecursionT(pColumn, parentPos, upper, daughterCell, currentLineage, lineageSizeIndex, data)
    # find lower daughter
    daughterCell.right = tryRecursionT(pColumn, lower, parentPos, daughterCell, currentLineage, lineageSizeIndex, data)

    # add daughter to current Lineage
    currentLineage.append(daughterCell)
    return daughterCell


def tryRecursionB(pColumn, lower, upper, parentCell, currentLineage, lineageSizeIndex, data):
    """
    Method for Bottom half of Lineage Tree
    """
    found = False
    # check if this is the last possible cell
    if pColumn + 3 >= lineageSizeIndex:
        return None
    # find branch within provided range
    pColumn += 3
    for parentPos in range(upper + 1, lower + 1):
        if not math.isnan(data[parentPos][pColumn]):
            found = True
            break
    if not found:
        return None
    # store values into lineage here
    daughterCell = c(parent=parentCell, gen=parentCell.gen + 1, synthetic=parentCell.synthetic)
    daughterCell.obs = [ 0, 0, 0, 0, data[parentPos][pColumn+2] ] # This stores the Time at cell division

    #[x  x] case
    if data[parentPos][pColumn] == data[parentPos][pColumn+2]:
        daughterCell.obs[0] = (data[parentPos][pColumn] == 145) #live/die G1
        daughterCell.obs[2] = data[parentPos][pColumn] - parentCell.obs[4]#Time Spent in G1
        daughterCell.obs[1] = None #Did not go to G2
        daughterCell.obs[3] = 0 #Spent no time in G2

    #[x  y]/[x y  ] case (general)
    else:
        #[1  y]/[1 y  ] case is not possible anymore
        daughterCell.obs[0] = True  #survived G1
        daughterCell.obs[2] = data[parentPos][pColumn] -  parentCell.obs[4]#Time spent in G1

        #[x  y] case (general)
        if math.isnan(data[parentPos][pColumn + 1]):
            daughterCell.obs[1] = True  #survived G2
            daughterCell.obs[3] = data[parentPos][pColumn+2]-data[parentPos][pColumn] #Time spent in G2
        #[x y  ] case
        else:
            daughterCell.obs[1] = False #died in G2
            daughterCell.obs[3] = data[parentPos][pColumn+1]-data[parentPos][pColumn] #Time spent in G2
            
    # find upper daughter
    daughterCell.left = tryRecursionB(pColumn, parentPos, upper, daughterCell, currentLineage, lineageSizeIndex, data)
    # find lower daughter
    daughterCell.right = tryRecursionB(pColumn, lower, parentPos, daughterCell, currentLineage, lineageSizeIndex, data)

    # add daughter to current Lineage
    currentLineage.append(daughterCell)
    return daughterCell
