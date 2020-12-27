""" This file exports excel sheets of lineages of cells with their states assigned to them after fitting. """

import numpy as np
import pandas as pd
import xlsxwriter

from .Analyze import Analyze_list
from .tHMM import tHMM
from .data.Lineage_collections import gemControl, gem5uM, Gem10uM, Gem30uM, Lapatinib_Control, Lapt25uM, Lapt50uM, Lap250uM, len_lp_cntr, len_lp_25, len_lp_50, len_lp_250, len_gm_cntr, len_gm_5, len_gm_10, len_gm_30

""" This is to run the tHMM objects 
# fitting gemc and lapat
gemm = [Lapatinib_Control + gemControl, gem5uM, Gem10uM, Gem30uM]
lptt = [Lapatinib_Control + gemControl, Lapt25uM, Lapt50uM, Lap250uM]

lapt_tHMMobj_list, lapt_states_list, _ = Analyze_list(lptt, 3, fpi=True)
gemc_tHMMobj_list, gemc_states_list, _ = Analyze_list(gemm, 4, fpi=True)

# assigning the estimated states to the cells
for idx, lapt_tHMMobj in enumerate(lapt_tHMMobj_list):
    for lin_indx, lin in enumerate(lapt_tHMMobj.X):
        for cell_indx, cell in enumerate(lin.output_lineage):
            cell.state = lapt_states_list[idx][lin_indx][cell_indx]

for idx, gemc_tHMMobj in enumerate(gemc_tHMMobj_list):
    for lin_indx, lin in enumerate(gemc_tHMMobj.X):
        for cell_indx, cell in enumerate(lin.output_lineage):
            cell.state = gemc_states_list[idx][lin_indx][cell_indx]
"""

def assign_states(input_X):
    """ Given a list of lineages, it returns a 2D array of cell states in the form of a tree that will then be written into excel sheets. """
    X = []
    for lin in input_X:
        # we will fill this 17 x 12 array in the form of a lineage with the state of cells and this array forms each lineage block in excel
        arr = np.empty((17, 12))
        arr[:] = np.nan
        arr[9, 2] = lin.output_lineage[0].state
        # gen 2
        if len(lin) >= 2:
            arr[5, 5] = lin.output_lineage[1].state
            arr[13, 5] = lin.output_lineage[2].state
            if len(lin.output_lineage) >= 4:
                # gen 3
                arr[15, 8] = lin.output_lineage[3].state
                arr[11, 8] = lin.output_lineage[4].state
                if len(lin.output_lineage) >= 6:
                    arr[7, 8] = lin.output_lineage[5].state
                    arr[3, 8] = lin.output_lineage[6].state
                    if len(lin.output_lineage) >= 8:
                        # gen 4
                        arr[16, 11] = lin.output_lineage[7].state
                        arr[14, 11] = lin.output_lineage[8].state
                        if len(lin.output_lineage) >= 10:
                            arr[12, 11] = lin.output_lineage[9].state
                            arr[10, 11] = lin.output_lineage[10].state
                            if len(lin.output_lineage) >= 12:
                                arr[8, 11] = lin.output_lineage[11].state
                                arr[6, 11] = lin.output_lineage[12].state
                                if len(lin.output_lineage) >= 14:
                                    arr[4, 11] = lin.output_lineage[13].state
                                    arr[2, 11] = lin.output_lineage[14].state
        X.append(arr)
    return X


def deintegrate(population, len_condition, gem=False):
    """ Given the tHMMob.X and the array that holds lengths of excel sheets for a condition, it returns a list including lineages corresponding to each excel sheet."""
    condition = []
    if gem:
        j = 4
        for lens in len_condition:
            condition.append(population[j: j + lens])
            j += lens
    else:
        j = 0
        for lens in len_condition:
            condition.append(population[j: j + lens])
            j += lens
    return condition


def write_onExcel(lapt_tHMMobj_list, len_lp_cntr, len_lp_25, len_lp_50, len_lp_250, conc1, conc2, conc3, gem=False):
    """ Write cell states with the pattern of input data (in a binary format) to excel sheets. """
    # each of these is a list holding populations that belong to each sheet we imported.
    if gem:
        lpcont = deintegrate(lapt_tHMMobj_list[0].X, len_lp_cntr, gem=True)
        cnt = "gmc_control"
    else:
        lpcont = deintegrate(lapt_tHMMobj_list[0].X, len_lp_cntr)
        cnt = "lpt_control"
    lp25 = deintegrate(lapt_tHMMobj_list[1].X, len_lp_25)
    lp50 = deintegrate(lapt_tHMMobj_list[2].X, len_lp_50)
    lp250 = deintegrate(lapt_tHMMobj_list[3].X, len_lp_250)

    # create 2D arrays that we will fill the excel sheets with them.
    # lapatinib
    lpt_cnt = [assign_states(i) for i in lpcont]
    lpt_25 = [assign_states(i) for i in lp25]
    lpt_50 = [assign_states(i) for i in lp50]
    lpt_250 = [assign_states(i) for i in lp250]

    for ind, sheet in enumerate(lpt_cnt):
        j = 1
        writer = pd.ExcelWriter(cnt + str(ind) + ".xlsx", engine='xlsxwriter')
        for arrays in sheet:
            df = pd.DataFrame(arrays)
            df.to_excel(writer, sheet_name='sheet1', startrow=j)
            j += 19
        writer.save()

    for ind, sheet in enumerate(lpt_25):
        j = 1
        writer = pd.ExcelWriter(conc1 + str(ind) + ".xlsx", engine='xlsxwriter')
        for arrays in sheet:
            df = pd.DataFrame(arrays)
            df.to_excel(writer, sheet_name='sheet1', startrow=j)
            j += 19
        writer.save()

    for ind, sheet in enumerate(lpt_50):
        j = 1
        writer = pd.ExcelWriter(conc2 + str(ind) + ".xlsx", engine='xlsxwriter')
        for arrays in sheet:
            df = pd.DataFrame(arrays)
            df.to_excel(writer, sheet_name='sheet1', startrow=j)
            j += 19
        writer.save()

    for ind, sheet in enumerate(lpt_250):
        j = 1
        writer = pd.ExcelWriter(conc3 + str(ind) + ".xlsx", engine='xlsxwriter')
        for arrays in sheet:
            df = pd.DataFrame(arrays)
            df.to_excel(writer, sheet_name='sheet1', startrow=j)
            j += 19
        writer.save()
