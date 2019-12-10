from multiprocessing import Process

from .figure2 import makeFigure2
from .figure8 import makeFigure8
from .figure9 import makeFigure9
from .figure11 import makeFigure11
from .figure12 import makeFigure12

def RunAll():
    p1 = Process(target=makeFigure2)
    p2 = Process(target=makeFigure8)
    p3 = Process(target=makeFigure9)
    p4 = Process(target=makeFigure11)
    p5 = Process(target=makeFigure12)

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()