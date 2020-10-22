""" This is only to run the export_script. """
import xlsxwriter
import pandas as pd

from .figureCommon import getSetup, subplotLabel
from ..export_states import lpt_25, lpt_50, lpt_250, gmc_5, gmc_10, gmc30

def makeFigure():
    """ makes figure. """
    ax, f = getSetup((3, 6), (2, 1))
    ##------------- writing into an excel sheet ------------------##

    # lapatinib
    for ind, sheet in enumerate(lpt_25):
        j = 1
        for idx, arrays in enumerate(sheet):
            df = pd.DataFrame(arrays)
            df.to_excel(excel_writer = "lpt_25_"+str(ind)+".xlsx", startrow=j)
            j += 19

    for ind, sheet in enumerate(lpt_50):
        j = 1
        for idx, arrays in enumerate(sheet):
            df = pd.DataFrame(arrays)
            df.to_excel(excel_writer = "lpt_50_"+str(ind)+".xlsx", startrow=j)
            j += 19

    for ind, sheet in enumerate(lpt_250):
        j = 1
        for idx, arrays in enumerate(sheet):
            df = pd.DataFrame(arrays)
            df.to_excel(excel_writer = "lpt_25_"+str(ind)+".xlsx", startrow=j)
            j += 19

    # gemcitabine
    for ind, sheet in enumerate(gmc_5):
        j = 1
        for idx, arrays in enumerate(sheet):
            df = pd.DataFrame(arrays)
            df.to_excel(excel_writer = "gmc_5_"+str(ind)+".xlsx", startrow=j)
            j += 19


    for ind, sheet in enumerate(gmc_10):
        j = 1
        for idx, arrays in enumerate(sheet):
            df = pd.DataFrame(arrays)
            df.to_excel(excel_writer = "lpt_25_"+str(ind)+".xlsx", startrow=j)
            j += 19


    for ind, sheet in enumerate(gmc_30):
        j = 1
        for idx, arrays in enumerate(sheet):
            df = pd.DataFrame(arrays)
            df.to_excel(excel_writer = "lpt_25_"+str(ind)+".xlsx", startrow=j)
            j += 19
    return f
