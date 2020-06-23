#!/usr/bin/env python3
from lineage.figures.figureCommon import overlayCartoon
import sys
import time
import matplotlib
import numpy as np
matplotlib.use('AGG')

fdir = './output/'
cartoon_dir = r"./lineage/figures/cartoons"

# TODO: Remove this one day.
np.random.seed(1)

if __name__ == '__main__':
    start = time.time()
    nameOut = 'figure' + sys.argv[1]

    exec('from lineage.figures import ' + nameOut)
    ff = eval(nameOut + '.makeFigure()')
    ff.savefig(fdir + nameOut + '.svg', dpi=ff.dpi, bbox_inches='tight', pad_inches=0)

    print(f'Figure {sys.argv[1]} is done after {time.time() - start} seconds.\n')

    if sys.argv[1] == 'S02':
        # Overlay Figure 2 cartoon
        overlayCartoon(fdir + 'figureS02.svg',
                       f'{cartoon_dir}/figure2.svg', 0, 0, scalee=0.2)

    if sys.argv[1] == 'S03':
        # Overlay Figure 3 cartoon
        overlayCartoon(fdir + 'figureS03.svg',
                       f'{cartoon_dir}/figure3.svg', 0, 0, scalee=0.18)

    if sys.argv[1] == 'S04':
        # Overlay Figure 4 cartoon
        overlayCartoon(fdir + 'figureS04.svg',
                       f'{cartoon_dir}/figure4.svg', 0, 0, scalee=0.18)

    if sys.argv[1] == 'S05':
        # Overlay Figure 5 cartoon
        overlayCartoon(fdir + 'figureS05.svg',
                       f'{cartoon_dir}/figure5.svg', 0, 0, scalee=0.18)

    if sys.argv[1] == 'S06':
        # Overlay Figure 6 cartoon
        overlayCartoon(fdir + 'figureS06.svg',
                       f'{cartoon_dir}/figure6.svg', 0, 0, scalee=0.19)

    if sys.argv[1] == 'S07':
        # Overlay Figure 7 cartoon
        overlayCartoon(fdir + 'figureS07.svg',
                       f'{cartoon_dir}/figure7.svg', 0, 0, scalee=0.18)

    if sys.argv[1] == 'S08':
        # Overlay Figure 8 cartoon
        overlayCartoon(fdir + 'figureS08.svg',
                       f'{cartoon_dir}/figure8.svg', 0, 0, scalee=0.18)

    if sys.argv[1] == 'S09':
        # Overlay Figure 9 cartoon
        overlayCartoon(fdir + 'figureS09.svg',
                       f'{cartoon_dir}/figure9.svg', 0, 0, scalee=0.18)
