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
                       f'{cartoon_dir}/figureS01.svg', 65, -10, scalee=0.62)

    if sys.argv[1] == 'S09':
        # Overlay Figure 9 cartoon
        overlayCartoon(fdir + 'figureS09.svg',
                       f'{cartoon_dir}/figureS02.svg', 80, 0, scalee=0.5)

    if sys.argv[1] == 'S10':
        # Overlay Figure 10 cartoon
        overlayCartoon(fdir + 'figureS10.svg',
                       f'{cartoon_dir}/figureS03.svg', 90, 5, scalee=0.57)

    if sys.argv[1] == '1':
        # Overlay Figure 1 cartoon
        overlayCartoon(fdir + 'figure1.svg',
                       f'{cartoon_dir}/xaxis-h.svg', 30, 196, scalee=1.1)
        overlayCartoon(fdir + 'figure1.svg',
                       f'{cartoon_dir}/xaxis-h.svg', 215, 196, scalee=1.1)

    if sys.argv[1] == '4':
        # Overlay Figure 4 cartoon
        overlayCartoon(fdir + 'figure4.svg',
                       f'{cartoon_dir}/experimentEndLine.svg', 590, 37, scalee=0.25)

    if sys.argv[1] == '5':
        # Overlay Figure 5 cartoon
        overlayCartoon(fdir + 'figure5.svg',
                       f'{cartoon_dir}/figure5.svg', 5, 5, scalee=1.55)

    if sys.argv[1] == '6':
        # Overlay Figure 6 cartoon
        overlayCartoon(fdir + 'figure6.svg',
                       f'{cartoon_dir}/figure6.svg', 150, 50, scalee=0.5)

    if sys.argv[1] == '11':
        # Overlay Transition block
        overlayCartoon(fdir + 'figure11.svg',
                       f'{cartoon_dir}/figure01.svg', 450, 50, scale_x=0.9, scale_y=1.2)
        overlayCartoon(fdir + 'figure11.svg',
                       f'{cartoon_dir}/lapatinib.svg', 5, 80, scalee=0.8)
        overlayCartoon(fdir + 'figure11.svg',
                       f'{cartoon_dir}/xaxis-h.svg', 450, 215, scalee=1.44)
        overlayCartoon(fdir + 'figure11.svg',
                       f'{cartoon_dir}/xaxis-h.svg', 645, 215, scalee=1.44)
        overlayCartoon(fdir + 'figure11.svg',
                       f'{cartoon_dir}/xaxis-h.svg', 840, 215, scalee=1.44)
        overlayCartoon(fdir + 'figure11.svg',
                       f'{cartoon_dir}/xaxis-h.svg', 1040, 215, scalee=1.44)

    if sys.argv[1] == '12':
        # Overlay Transition block
        overlayCartoon(fdir + 'figure12.svg',
                       f'{cartoon_dir}/figure02.svg', 450, 50, scale_x=0.9, scale_y=1.2)
        overlayCartoon(fdir + 'figure12.svg',
                       f'{cartoon_dir}/gemcitabine.svg', 10, 80, scalee=1.1)
        overlayCartoon(fdir + 'figure12.svg',
                       f'{cartoon_dir}/xaxis-h.svg', 450, 215, scalee=1.44)
        overlayCartoon(fdir + 'figure12.svg',
                       f'{cartoon_dir}/xaxis-h.svg', 645, 215, scalee=1.44)
        overlayCartoon(fdir + 'figure12.svg',
                       f'{cartoon_dir}/xaxis-h.svg', 840, 215, scalee=1.44)
        overlayCartoon(fdir + 'figure12.svg',
                       f'{cartoon_dir}/xaxis-h.svg', 1040, 215, scalee=1.44)

    if sys.argv[1] == 'S11':
        # Overlay Transition block
        overlayCartoon(fdir + 'figureS11.svg',
                       f'{cartoon_dir}/figure10.svg', 0, 20, scalee=0.2, scale_x=1.6, scale_y=1.2)
        overlayCartoon(fdir + 'figureS11.svg',
                       f'{cartoon_dir}/figure101.svg', 180, 20, scalee=0.2, scale_x=1.6, scale_y=1.2)
        overlayCartoon(fdir + 'figureS11.svg',
                       f'{cartoon_dir}/figure102.svg', 360, 20, scalee=0.2, scale_x=1.6, scale_y=1.2)
        overlayCartoon(fdir + 'figureS11.svg',
                       f'{cartoon_dir}/figure103.svg', 540, 20, scalee=0.2, scale_x=1.6, scale_y=1.2)

    if sys.argv[1] == 'S12':
        # Overlay Transition block
        overlayCartoon(fdir + 'figureS12.svg',
                       f'{cartoon_dir}/figure15.svg', 0, 20, scalee=0.2, scale_x=1.6, scale_y=1.2)
        overlayCartoon(fdir + 'figureS12.svg',
                       f'{cartoon_dir}/figure151.svg', 180, 20, scalee=0.2, scale_x=1.6, scale_y=1.2)
        overlayCartoon(fdir + 'figureS12.svg',
                       f'{cartoon_dir}/figure152.svg', 360, 20, scalee=0.2, scale_x=1.6, scale_y=1.2)
        overlayCartoon(fdir + 'figureS12.svg',
                       f'{cartoon_dir}/figure153.svg', 540, 20, scalee=0.2, scale_x=1.6, scale_y=1.2)

    if sys.argv[1] == 'S13':
        # Overlay Transition block
        overlayCartoon(fdir + 'figureS13.svg',
                       f'{cartoon_dir}/figure16.svg', 0, 20, scalee=0.8, scale_y=0.5)
        overlayCartoon(fdir + 'figureS13.svg',
                       f'{cartoon_dir}/figure161.svg', 200, 20, scalee=0.8, scale_y=0.5)

    if sys.argv[1] == 'S14':
        # Overlay Transition block
        overlayCartoon(fdir + 'figureS14.svg',
                       f'{cartoon_dir}/figure17.svg', 0, 20, scalee=0.8, scale_y=0.5)
        overlayCartoon(fdir + 'figureS14.svg',
                       f'{cartoon_dir}/figure171.svg', 200, 20, scalee=0.8, scale_y=0.5)
