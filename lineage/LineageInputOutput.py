""" The file contains the methods used to input lineage data from the Heiser lab. """

import pandas as pd
import math
from CellVar import CellVar as c, double


def import_Heiser(path=r"~/Projects/CAPSTONE/lineage/data/heiser_data/LT_AU003_A3_4_Lapatinib_V2.xlsx"):
    excel_file = pd.read_excel(path, header=None)
    data = excel_file.to_numpy()
    # Cell.obs stored as [G1, G2/S (nan if cell dies), Death (nan if cell does not die)]
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
            parentCell.obs.append(data[lPos][1])
            parentCell.obs.append(data[lPos][1 + 2])
            parentCell.obs.append(data[lPos][1 + 1])
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

            # add first generation to lineage AFTER all left/right are defined properly (python passes copies to append)
            currentLineage.append(parentCell)

            # store lineage in list of lineages
            lineages.append(currentLineage)
    return lineages


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
    daughterCell.obs.append(data[parentPos][pColumn])
    daughterCell.obs.append(data[parentPos][pColumn + 2])
    daughterCell.obs.append(data[parentPos][pColumn + 1])
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
    daughterCell.obs.append(data[parentPos][pColumn])
    daughterCell.obs.append(data[parentPos][pColumn + 2])
    daughterCell.obs.append(data[parentPos][pColumn + 1])
    # find upper daughter
    daughterCell.left = tryRecursionB(pColumn, parentPos, upper, daughterCell, currentLineage, lineageSizeIndex, data)
    # find lower daughter
    daughterCell.right = tryRecursionB(pColumn, lower, parentPos, daughterCell, currentLineage, lineageSizeIndex, data)

    # add daughter to current Lineage
    currentLineage.append(daughterCell)
    return daughterCell
