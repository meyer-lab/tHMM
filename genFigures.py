#!/usr/bin/env python3
from lineage.figures.figureCommon import overlayCartoon
import sys
import time
import matplotlib
import numpy as np
matplotlib.use('AGG')

fdir = './output/'
cartoon_dir = r"./lineage/figures/cartoons"

if __name__ == '__main__':
    start = time.time()
    nameOut = 'figure' + sys.argv[1]

    exec('from lineage.figures import ' + nameOut)
    ff = eval(nameOut + '.makeFigure()')
    ff.savefig(fdir + nameOut + '.svg', dpi=ff.dpi,
               bbox_inches='tight', pad_inches=0)

    print(
        f'Figure {sys.argv[1]} is done after {time.time() - start} seconds.\n')

    if sys.argv[1] == 'S01':
        # Overlay Figure 1 cartoon
        overlayCartoon(fdir + 'figureS01.svg',
                       f'{cartoon_dir}/figureS01.svg', 50, 0, scalee=0.78)

    if sys.argv[1] == 'S02':
        # Overlay Figure 2 cartoon
        overlayCartoon(fdir + 'figureS02.svg',
                       f'{cartoon_dir}/figureS02.svg', 80, 0, scalee=0.7)

    if sys.argv[1] == 'S03':
        # Overlay Figure 3 cartoon
        overlayCartoon(fdir + 'figureS03.svg',
                       f'{cartoon_dir}/figureS03.svg', 90, 5, scalee=0.67)

    if sys.argv[1] == 'S04':
        # Overlay Figure 4 cartoon
        overlayCartoon(fdir + 'figureS04.svg',
                       f'{cartoon_dir}/figureS04.svg', 15, 10, scalee=0.54)

    if sys.argv[1] == 'S05':
        # Overlay Figure 5 cartoon
        overlayCartoon(fdir + 'figureS05.svg',
                       f'{cartoon_dir}/figureS05.svg', 15, 10, scalee=0.54)

    if sys.argv[1] == 'S06':
        # Overlay Figure 6 cartoon
        overlayCartoon(fdir + 'figureS06.svg',
                       f'{cartoon_dir}/figureS06.svg', 30, 0, scalee=0.53)

    if sys.argv[1] == 'S07':
        # Overlay Figure 7 cartoon
        overlayCartoon(fdir + 'figureS07.svg',
                       f'{cartoon_dir}/figureS07.svg', 30, 0, scalee=0.53)

    if sys.argv[1] == 'S08':
        # Overlay Figure 8 cartoon
        overlayCartoon(fdir + 'figureS08.svg',
                       f'{cartoon_dir}/figureS01.svg', 45, -10, scalee=0.82)

    if sys.argv[1] == 'S09':
        # Overlay Figure 9 cartoon
        overlayCartoon(fdir + 'figureS09.svg',
                       f'{cartoon_dir}/figureS02.svg', 80, 0, scalee=0.7)

    if sys.argv[1] == 'S10':
        # Overlay Figure 10 cartoon
        overlayCartoon(fdir + 'figureS10.svg',
                       f'{cartoon_dir}/figureS03.svg', 90, 5, scalee=0.67)

    if sys.argv[1] == '1':
        # Overlay Figure 1 cartoon
        overlayCartoon(fdir + 'figure1.svg',
                       f'{cartoon_dir}/xaxis-h.svg', 30, 110, scalee=1.1)
        overlayCartoon(fdir + 'figure1.svg',
                       f'{cartoon_dir}/xaxis-h.svg', 215, 110, scalee=1.1)

    if sys.argv[1] == '5':
        # Overlay Figure 5 cartoon
        overlayCartoon(fdir + 'figure5.svg',
                       f'{cartoon_dir}/figure5.svg', 10, 5, scalee=0.75)

    if sys.argv[1] == '6':
        # Overlay Figure 6 cartoon
        overlayCartoon(fdir + 'figure6.svg',
                       f'{cartoon_dir}/figure6.svg', 70, 0, scalee=0.65)

    if sys.argv[1] == '7':
        # Overlay Figure 7 cartoon
        overlayCartoon(fdir + 'figure7.svg',
                       f'{cartoon_dir}/figure7.svg', 60, 10, scalee=0.85)
