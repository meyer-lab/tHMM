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

    # a = pd.DataFrame(columns=["state", "length"])
    a = [0, 0, 2, 3, 2, 1, 1, 3, 3, 4]
    b = [np.nan, 10, -20, 30, np.nan, 50, 60, np.nan, 70, 80]
    c = ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"]
    sns.stripplot(x=a, y=b, hue=c, ax=ax[0])


    return f
