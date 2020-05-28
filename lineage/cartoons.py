#!/usr/bin/env python3

from ckine.figures.figureCommon import overlayCartoon
import sys
import logging
import time
import matplotlib
matplotlib.use('AGG')

fdir = './Manuscript/Figures/'

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

if __name__ == '__main__':
    start = time.time()
    nameOut = 'figure' + sys.argv[1]

    exec('from ckine.figures import ' + nameOut)
    ff = eval(nameOut + '.makeFigure()')
    ff.savefig(fdir + nameOut + '.svg', dpi=ff.dpi, bbox_inches='tight', pad_inches=0)

    if sys.argv[1] == '1':
        # Overlay Figure 1 cartoon
        overlayCartoon(fdir + 'figure1.svg',
                       './graph_all.svg', 1, 12, scalee=0.33, scale_x=1.07, scale_y=0.94)  # scalee was 0.35, y was 15
        overlayCartoon(fdir + 'figure1.svg',
                       './ckine/data/cell_legend.svg', 375, 105, scalee=0.14)  # scalee was 0.35, y was 15

    if sys.argv[1] == '2':
        # Overlay Figure 2 cartoon
        overlayCartoon(fdir + 'figure2.svg',
                       './ckine/data/simple_crosstalk.svg', 29, 0, scalee=0.1)  # might need to adjust this

    if sys.argv[1] == '3':
        # Overlay Figure 3 cartoon
        overlayCartoon(fdir + 'figure3.svg',
                       './ckine/data/tensor3D.svg', 16, 160, scalee=1.0)

    logging.info('%s is done after %s seconds.', nameOut, time.time() - start)