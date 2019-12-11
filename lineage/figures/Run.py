from multiprocessing import Pool
import sys
import matplotlib
matplotlib.use('AGG')

from .figure2 import makeFigure1
from .figure8 import makeFigure2
from .figure9 import makeFigure3
from .figure11 import makeFigure4
from .figure12 import makeFigure5

def Run():
    fdir = './output/'
    pool = multiprocessing.Pool( args.numProcessors )

    for i in range(1,5):
        result = pool.apply_async(eval('makeFigure'+ str(i)))
        result.savefig(fdir + 'figure' + str(i) + '.svg', dpi=result.dpi, bbox_inches='tight', pad_inches=0)

        print('figure' + str(i) + ' is done.')

# def Run():
    
#     pool = multiprocessing.Pool( args.numProcessors )
    
#     p1 = Process(target=makeFigure2)
#     p2 = Process(target=makeFigure8)
#     p3 = Process(target=makeFigure9)
#     p4 = Process(target=makeFigure11)
#     p5 = Process(target=makeFigure12)

#     P = [p1, p2, p3, p4, p5]
#     p1.start()
#     print("starting p1")
#     p2.start()
#     print("starting p2")
#     p3.start()
#     print("starting p3")
#     p4.start()
#     print("starting p4")
#     p5.start()
#     print("starting p5")
