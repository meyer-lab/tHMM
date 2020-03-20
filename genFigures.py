#!/usr/bin/env python3

import sys
import time
import matplotlib
import numpy as np
matplotlib.use('AGG')

fdir = './output/'

# TODO: Remove this one day.
np.random.seed(1)

if __name__ == '__main__':
    start = time.time()
    nameOut = 'figure' + sys.argv[1]

    exec('from lineage.figures import ' + nameOut)
    ff = eval(nameOut + '.makeFigure()')
    ff.savefig(fdir + nameOut + '.svg', dpi=ff.dpi, bbox_inches='tight', pad_inches=0)

    print(f'\n{nameOut} is done after {time.time() - start} seconds.\n')
