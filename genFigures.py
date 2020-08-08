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

    if sys.argv[1] == 'S02':
        # Overlay Figure 2 cartoon
        overlayCartoon(fdir + 'figureS02.svg',
                       f'{cartoon_dir}/figureS02.svg', 50, 0, scalee=0.75)

    if sys.argv[1] == 'S03':
        # Overlay Figure 3 cartoon
        overlayCartoon(fdir + 'figureS03.svg',
                       f'{cartoon_dir}/figureS03.svg', 80, 0, scalee=0.7)

    if sys.argv[1] == 'S04':
        # Overlay Figure 4 cartoon
        overlayCartoon(fdir + 'figureS04.svg',
                       f'{cartoon_dir}/figureS04.svg', 60, -10, scalee=0.8)

    if sys.argv[1] == 'S05':
        # Overlay Figure 5 cartoon
        overlayCartoon(fdir + 'figureS05.svg',
                       f'{cartoon_dir}/figureS05.svg', 60, 0, scalee=0.7)

    if sys.argv[1] == 'S06':
        # Overlay Figure 6 cartoon
        overlayCartoon(fdir + 'figureS06.svg',
                       f'{cartoon_dir}/figureS06.svg', 10, 0, scalee=0.38)

    if sys.argv[1] == 'S07':
        # Overlay Figure 7 cartoon
        overlayCartoon(fdir + 'figureS07.svg',
                       f'{cartoon_dir}/figureS07.svg', 10, -10, scalee=0.38)

    if sys.argv[1] == 'S08':
        # Overlay Figure 8 cartoon
        overlayCartoon(fdir + 'figureS08.svg',
                       f'{cartoon_dir}/figureS08.svg', 40, 0, scalee=0.35)

    if sys.argv[1] == 'S09':
        # Overlay Figure 9 cartoon
        overlayCartoon(fdir + 'figureS09.svg',
                       f'{cartoon_dir}/figureS09.svg', 45, 0, scalee=0.32)

    if sys.argv[1] == 'S12':
        # Overlay Figure 12 cartoon
        overlayCartoon(fdir + 'figureS12.svg',
                       f'{cartoon_dir}/figureS02.svg', 0, 10, scalee=0.63)

    if sys.argv[1] == 'S13':
        # Overlay Figure 13 cartoon
        overlayCartoon(fdir + 'figureS13.svg',
                       f'{cartoon_dir}/figureS03.svg', 80, 0, scalee=0.7)

    if sys.argv[1] == 'S14':
        # Overlay Figure 14 cartoon
        overlayCartoon(fdir + 'figureS14.svg',
                       f'{cartoon_dir}/figureS04.svg', 0, 10, scalee=0.64)

    if sys.argv[1] == 'S15':
        # Overlay Figure 15 cartoon
        overlayCartoon(fdir + 'figureS15.svg',
                       f'{cartoon_dir}/figureS05.svg', 60, 0, scalee=0.7)

    if sys.argv[1] == 'S22':
        # Overlay Figure 22 cartoon
        overlayCartoon(fdir + 'figureS22.svg',
                       f'{cartoon_dir}/figureS02.svg', 50, 0, scalee=0.75)

    if sys.argv[1] == 'S23':
        # Overlay Figure 23 cartoon
        overlayCartoon(fdir + 'figureS23.svg',
                       f'{cartoon_dir}/figureS03.svg', 80, 0, scalee=0.7)

    if sys.argv[1] == 'S24':
        # Overlay Figure 24 cartoon
        overlayCartoon(fdir + 'figureS24.svg',
                       f'{cartoon_dir}/figureS04.svg', 60, -10, scalee=0.8)

    if sys.argv[1] == 'S25':
        # Overlay Figure 25 cartoon
        overlayCartoon(fdir + 'figureS25.svg',
                       f'{cartoon_dir}/figureS05.svg', 60, 0, scalee=0.7)

    if sys.argv[1] == 'S52':
        # Overlay Figure 2 cartoon
        overlayCartoon(fdir + 'figureS52.svg',
                       f'{cartoon_dir}/figure2.svg', 80, -20, scalee=0.85)

    if sys.argv[1] == '1':
        # Overlay Figure 1 cartoon
        overlayCartoon(fdir + 'figure1.svg',
                       f'{cartoon_dir}/figure1a.svg', 10, 0, scalee=0.48)
        overlayCartoon(fdir + 'figure1.svg',
                       f'{cartoon_dir}/figure1b.svg', 10, 120, scalee=0.48)
        overlayCartoon(fdir + 'figure1.svg',
                       f'{cartoon_dir}/figure1c.svg', 10, 240, scalee=0.48)
        overlayCartoon(fdir + 'figure1.svg',
                       f'{cartoon_dir}/figure1d.svg', 165, 0, scalee=0.48)
        overlayCartoon(fdir + 'figure1.svg',
                       f'{cartoon_dir}/figure1e.svg', 165, 120, scalee=0.48)
        overlayCartoon(fdir + 'figure1.svg',
                       f'{cartoon_dir}/figure1f.svg', 165, 240, scalee=0.48)
        overlayCartoon(fdir + 'figure1.svg',
                       f'{cartoon_dir}/xaxis.svg', 10, 360, scalee=1.28)
        overlayCartoon(fdir + 'figure1.svg',
                       f'{cartoon_dir}/xaxis.svg', 165, 360, scalee=1.28)

    if sys.argv[1] == '2':
        # Overlay Figure 2 cartoon
        overlayCartoon(fdir + 'figure2.svg',
                       f'{cartoon_dir}/figure2.svg', 0, 0, scalee=0.098)

    if sys.argv[1] == '3':
        # Overlay Figure 3 cartoon
        overlayCartoon(fdir + 'figure3.svg',
                       f'{cartoon_dir}/lineage_notcen.svg', 20, 20, scalee=0.4)
        overlayCartoon(fdir + 'figure3.svg',
                       f'{cartoon_dir}/lineage_cen.svg', 190, 20, scalee=0.4)

    if sys.argv[1] == '4':
        # Overlay Figure 4 cartoon
        overlayCartoon(fdir + 'figure4.svg',
                       f'{cartoon_dir}/figure4.svg', 0, 15, scalee=0.46)

    if sys.argv[1] == '5':
        # Overlay Figure 5 cartoon
        overlayCartoon(fdir + 'figure5.svg',
                       f'{cartoon_dir}/figure5.svg', 35, 0, scalee=0.65)

    if sys.argv[1] == '6':
        # Overlay Figure 6 cartoon
        overlayCartoon(fdir + 'figure6.svg',
                       f'{cartoon_dir}/figure6.svg', 0, 20, scalee=0.85)
