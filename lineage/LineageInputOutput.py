""" The file contains the methods used to input lineage data from the Heiser lab. """
import math
import pandas as pd
from .CellVar import CellVar as c
import numpy as np


def import_Heiser(path):
    global_exp_time = [-1]
    """
    Imports data from the Heiser lab
    Outputs a list of lists containing cells containing observations from the Excel file
    In this particular dataset, we look at the following phenotypes:

    1. survival past G1 (death or transition into G2),
    2. survival past G2 (death or division into daughter cells),
    3. time spent in G1,
    4. time spent in G2,
    5. G1 time censorship,
    6. G2 time censorship
    """
    excel_file = pd.read_excel(path, header=None)
    data = excel_file.to_numpy()

    # Checking if exp_time was added to the file. The function will work even if exp_time isn't set, but may need proofreading
    if "exp_time" in data[0]:
        exp_time = data[1][np.where(data[0] == "exp_time")[0][0]]
        global_exp_time = [exp_time]
    else:
        print("exp_time has not been added to this file")
    # current Lineage Posistion
    lPos = 0
    # current Lineage Number
    lineageNo = 0
    # Next Upper range value
    nextUp = 1

    # find Lineages
    lineages = []
    while lPos < len(data):
        if "exp_time" not in data[0]:
            exp_time = -1
        lPos += 1
        # find Next Lineage Position
        while lPos < len(data) and math.isnan(data[lPos][0]):
            lPos += 1

        # checking if lineage # lines up
        if lPos < len(data):
            lineageNo += 1
            assert lineageNo == data[lPos][0]

        # determine if lineage has cells
        if lPos < len(data) and not math.isnan(data[lPos][1]):
            # Check that there is a value
            assert not math.isnan(
                data[lPos][2]) or not math.isnan(data[lPos][3]), f"Value missing in first cell of lineage {lineageNo}"

            # add list for the lineage
            currentLineage = []
            # make Parent
            parentCell = c(parent=None, gen=1, synthetic=False)
            divisionTime = data[lPos][1 + 2]
            parentCell.obs = [0, 0, 0, 0, 0, 0]

            # find lower value of range and store next upper
            upper = nextUp
            nextUp = lPos + 10
            if nextUp >= len(data):
                lower = len(data)
            else:
                # checking that spacing is correct
                assert not math.isnan(
                    data[nextUp + 8][0]), "File is improperly formatted (lineages spaced differently)"
                lower = nextUp - 2

            # find upper daughter and recurse
            parentCell.left = tryRecursion(
                1, lPos, upper, parentCell, currentLineage, data, divisionTime, exp_time, global_exp_time)
            # find lower daughter and recurse
            parentCell.right = tryRecursion(
                1, lower, lPos, parentCell, currentLineage, data, divisionTime, exp_time, global_exp_time)

            # This will only run if exp_time has not been put into the file
            if exp_time == -1 and not math.isnan(data[lPos][1 + 2]) and parentCell.left is None and parentCell.right is None:
                exp_time = data[lPos][1 + 2]
                if global_exp_time[0] != -1:
                    assert exp_time == global_exp_time[0], f"Exp_time discrepancy in file {exp_time} and {global_exp_time[0]}"
            if global_exp_time[0] == -1 and exp_time != -1:
                global_exp_time[0] = exp_time

            # [exp_time  exp_time] case
            if data[lPos][1] == data[lPos][1 + 2]:
                # check that they are both exp_time (non exp_time should not exist)
                assert data[lPos][1] == exp_time, '[x  _  x] case where x =/= exp time'
                parentCell.obs[0] = float("nan") if (
                    data[lPos][1] == exp_time) else 0  # live/die G1
                parentCell.obs[1] = float("nan")  # Did not go to G2
                parentCell.obs[2] = data[lPos][1]  # Time Spent in G1
                parentCell.obs[3] = float("nan")  # Spent no time in G2
                parentCell.obs[4] = 0  # G1 is always censored for the first cell
                parentCell.obs[5] = float("nan")  # G2 outcome unknown

            # [x  y]/[x y  ] case (general)
            else:
                # [x x  ] case
                if data[lPos][1] == data[lPos][2]:
                    parentCell.obs[0] = 0  # Cell does not survive G1
                    parentCell.obs[2] = data[lPos][1]  # Time spent in G1
                    parentCell.obs[4] = 1  # G1 is not time censored
                    parentCell.obs[1] = float("nan")  # No info G2
                    parentCell.obs[3] = float("nan")  # No info G2
                    parentCell.obs[5] = float("nan")  # No info G2
                else:
                    # [1  y]/[1 y  ] case
                    if data[lPos][1] == 1:
                        # did not start in G1, but did transition
                        parentCell.obs[0] = 1  # survived G1
                        parentCell.obs[2] = float("nan")  # Spent no time in G1
                        parentCell.obs[4] = float("nan")  # G1 outcome unknown

                    # [x=/=1   y]/[x=/=1 y  ] case
                    else:
                        parentCell.obs[0] = 1  # survived G1
                        parentCell.obs[2] = data[lPos][1]  # Time spent in G1
                        # G1 is always censored in the first cell
                        parentCell.obs[4] = 0

                    # [x  y] case (general)
                    if math.isnan(data[lPos][1 + 1]):
                        parentCell.obs[1] = float("nan") if (
                            data[lPos][1 + 2] == exp_time) else 1  # survived/did not observe G2
                        parentCell.obs[3] = data[lPos][1 + 2] if (math.isnan(
                            parentCell.obs[2])) else data[lPos][1 + 2] - parentCell.obs[2]  # Time spent in G2

                    # [x y  ] case
                    else:
                        parentCell.obs[1] = 0  # died in G2
                        parentCell.obs[3] = data[lPos][1 + 1] if (math.isnan(
                            parentCell.obs[2])) else data[lPos][1 + 1] - parentCell.obs[2]  # Time spent in G2

                    # Time Censored Case  [x  exp_time]
                    # Censored if the cell started in G2 or total time is exp_time
                    parentCell.obs[5] = 0 if (
                        data[lPos][1 + 2] == exp_time or data[lPos][1] == 1) else 1

            # check that if there is one daughter there are both
            if parentCell.left is None or parentCell.right is None:
                assert parentCell.left is None and parentCell.right is None, f'Only one cell after division detected row {lPos+1}, column 2 of sheet' 
            #check that the cell did not divide if the cell is dead
            if parentCell.obs[0] == 0 or parentCell.obs[1] == 0:
                assert parentCell.left is None and parentCell.right is None, f'Cell death in row {lPos+1}, column 2 of sheet, but daughters were found'   
            # check all time values end up positive
            if not math.isnan(parentCell.obs[2]):
                assert parentCell.obs[2] >= 0, f"negative time value encountered, row {lPos+1}, column 2"
            if not math.isnan(parentCell.obs[3]):
                assert parentCell.obs[3] >= 0, f"negative time value encountered, row {lPos+1}, column 2"
            currentLineage.append(parentCell)
            # store lineage in list of lineages
            lineages.append(currentLineage)
    return lineages


def tryRecursion(pColumn, lower, upper, parentCell, currentLineage, data, divisionTime, exp_time, global_exp_time):
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
    if pColumn + 3 >= np.where(data[0] == "Lineage Size")[0]:
        return None

    # find branch within provided range
    pColumn += 3
    for parentPos in range(upper, lower):
        if not math.isnan(data[parentPos][pColumn]):
            found = True
            break
    if not found:
        return None


    # Check that the parent cell didn't get time censored (Likely divided in last frame)
    if divisionTime == exp_time:
        print(f'Cell time censorship, but daughters were found in row {parentPos+1}, column {pColumn+1}. By default they will be set to None')
        return None


    # Check that there is a value
    assert not math.isnan(data[parentPos][pColumn + 1]) or not math.isnan(
        data[parentPos][pColumn + 2]), f"Value missing in cell"

    # Creating daughter
    daughterCell = c(parent=parentCell, gen=parentCell.gen +
                     1, synthetic=parentCell.synthetic)
    daughterCell.obs = [0, 0, 0, 0, 0, 0]

    # find upper daughter
    daughterCell.left = tryRecursion(pColumn, parentPos, upper, daughterCell,
                                     currentLineage, data, data[parentPos][pColumn + 2], exp_time, global_exp_time)
    # find lower daughter
    daughterCell.right = tryRecursion(pColumn, lower, parentPos, daughterCell,
                                      currentLineage, data, data[parentPos][pColumn + 2], exp_time, global_exp_time)

    # This will only run if exp_time has not been put into the file
    if exp_time == -1 and not math.isnan(data[parentPos][pColumn + 2]) and daughterCell.left is None and daughterCell.right is None:
        exp_time = data[parentPos][pColumn + 2]
        if global_exp_time[0] != -1:
            assert exp_time == global_exp_time[0], f"Exp_time discrepancy in file {exp_time} and {global_exp_time[0]}"
    if global_exp_time[0] == -1 and exp_time != -1:
        global_exp_time[0] = exp_time

    # [x  x] case
    if data[parentPos][pColumn] == data[parentPos][pColumn + 2]:
        # Time Censored [exp_time  exp_time]
        assert data[parentPos][pColumn] == exp_time, '[x  _  x] case where x =/= exp time'
        if data[parentPos][pColumn] == exp_time:
            # We don't know the outcome of G1
            daughterCell.obs[0] = float("nan")
            daughterCell.obs[4] = 0  # G1 censored

        # Not Time Censored [x=/=exp_time   x=/=exp_time]
        # This Should not happen
        else:
            daughterCell.obs[0] = 0  # G1 death
            daughterCell.obs[4] = 1  # G1 uncensored

        daughterCell.obs[1] = float("nan")  # Did not go to G2
        daughterCell.obs[2] = data[parentPos][pColumn] - \
            divisionTime  # Time Spent in G1
        daughterCell.obs[3] = float("nan")  # Spent no time in G2
        daughterCell.obs[5] = float("nan")  # We don't have information about G2

    # [x  y]/[x y  ] case (general)
    else:
        # [x x  ] case
        if data[parentPos][pColumn + 1] == data[parentPos][pColumn]:
            daughterCell.obs[0] = 0
            daughterCell.obs[2] = data[parentPos][pColumn] - \
                divisionTime  # Time spent in G1
            daughterCell.obs[4] = 1
            daughterCell.obs[1] = float("nan")  # No info G2
            daughterCell.obs[3] = float("nan")  # No info G2
            daughterCell.obs[5] = float("nan")  # No info G2

        else:
            # [1  y]/[1 y  ] case is not possible anymore
            daughterCell.obs[0] = 1  # survived G1
            daughterCell.obs[2] = data[parentPos][pColumn] - \
                divisionTime  # Time spent in G1
            daughterCell.obs[4] = 1  # G1 uncensored

            # [x  y] case (general)
            if math.isnan(data[parentPos][pColumn + 1]):
                daughterCell.obs[1] = float("nan") if (
                    data[parentPos][pColumn + 2] == exp_time) else 1  # survived G2
                daughterCell.obs[3] = data[parentPos][pColumn + 2] - \
                    data[parentPos][pColumn]  # Time spent in G2
            # [x y  ] case
            else:
                daughterCell.obs[1] = 0  # died in G2
                daughterCell.obs[3] = data[parentPos][pColumn + 1] - \
                    data[parentPos][pColumn]  # Time spent in G2
            # Time Censored Case ([x  exp_time])
            # Censored if final time is exp_time, otherise uncensored
            daughterCell.obs[5] = 0 if (
                data[parentPos][pColumn + 2] == exp_time) else 1

    # check that if there is one daughter there are both
    if daughterCell.left is None or daughterCell.right is None:
        assert daughterCell.left is None and daughterCell.right is None, f'Only one cell after division detected row {parentPos+1}, column {pColumn+1} of sheet' 
    # check that the cell did not divide if the cell is dead
    if daughterCell.obs[0] == 0 or daughterCell.obs[1] == 0:
        assert daughterCell.left is None and daughterCell.right is None, f'Cell death in row {parentPos+1}, column {pColumn+1} of sheet, but daughters were found'   
    # check all time values end up positive
    if not math.isnan(daughterCell.obs[2]):
        assert daughterCell.obs[2] >= 0, f"negative time value encountered, row {parentPos+1}, column {pColumn+1}"
    if not math.isnan(daughterCell.obs[3]):
        assert daughterCell.obs[3] >= 0, f"negative time value encountered, row {parentPos+1}, column {pColumn+1}"
    # add daughter to current Lineage
    currentLineage.append(daughterCell)
    return daughterCell
