#!/usr/bin/env python3

import sys
import matplotlib
import numpy as np
import time
matplotlib.use('AGG')

fdir = './output/'

# TODO: Remove this one day.
np.random.seed(1)

if __name__ == '__main__':
    nameOut = 'figure' + sys.argv[1]

    exec('from lineage.figures import ' + nameOut)
    start_time = time.time()
    ff = eval(nameOut + '.makeFigure()')
    print("--- %s seconds ---" % (time.time() - start_time))
    ff.savefig(fdir + nameOut + '.svg', dpi=ff.dpi, bbox_inches='tight', pad_inches=0)

    print(nameOut + ' is done.')
