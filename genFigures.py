#!/usr/bin/env python3

import sys
import matplotlib
matplotlib.use('AGG')

fdir = './manuscript/figures/'


if __name__ == '__main__':
    nameOut = 'figure' + sys.argv[1]

    exec('from lineage.figures import ' + nameOut)
    ff = eval(nameOut + '.makeFigure()')
    ff.savefig(fdir + nameOut + '.svg', dpi=ff.dpi, bbox_inches='tight', pad_inches=0)

    print(nameOut + ' is done.')
