import numpy as np
import pandas as pd
import seaborn as sns

from .figureCommon import getSetup, subplotLabel

def makeFigure():
    """
    Makes figure 0.
    """
    ax, f = getSetup((4,4), (1, 1))
    subplotLabel(ax)

    a = pd.DataFrame(columns=["state", "length"])
    a["state"] = [0, 0, 2, 3, 2, 1, 1, 3, 3, 4]
    a["length"] = [np.nan, 10, -20, 30, np.nan, 50, 60, np.nan, 70, 80]

    sns.stripplot(x="state", y="length", data=a, ax=ax[0])


    return f
