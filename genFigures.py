#!/usr/bin/env python3

from lineage.figures.figure12 import makeFigure5
from lineage.figures.figure11 import makeFigure4
from lineage.figures.figure9 import makeFigure3
from lineage.figures.figure8 import makeFigure2
from lineage.figures.figure2 import makeFigure1
import multiprocessing
import sys
import matplotlib
matplotlib.use('AGG')


fdir = './output/'
pool = multiprocessing.Pool(multiprocessing.cpu_count())

if __name__ == '__main__':
    nameOut = 'makeFigure' + sys.argv[1]
    result = pool.apply_async(eval(nameOut))
    ff = result.get()
    ff.savefig(fdir + 'figure' + sys.argv[1] + '.svg', dpi=ff.dpi, bbox_inches='tight', pad_inches=0)

    print('figure' + sys.argv[1] + ' is done.')

# import sys
# import matplotlib
# matplotlib.use('AGG')

# fdir = './output/'


# if __name__ == '__main__':
#     nameOut = 'figure' + sys.argv[1]

#     exec('from lineage.figures import ' + nameOut)
#     ff = eval(nameOut + '.makeFigure()')
#     ff.savefig(fdir + nameOut + '.svg', dpi=ff.dpi, bbox_inches='tight', pad_inches=0)

#     print(nameOut + ' is done.')
