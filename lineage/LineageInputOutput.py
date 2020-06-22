""" The file contains the methods used to input lineage data from the Heiser lab. """

import pandas as pd


def import_Heiser(path=r"~/Projects/CAPSTONE/lineage/data/heiser_data/LT_AU003_A3_4_Lapatinib_V2.xlsx"):
    excel_file = pd.read_excel(path)
    data = excel_file.to_numpy()
    assert data[0][16] == "Lineage Size"
    