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
    
    if sys.argv[1] == '2':
        # Overlay Figure 2 cartoon
        overlayCartoon(fdir + 'figure2.svg',
                       f'{cartoon_dir}/figure2.svg', 70, 0, scalee=0.3) 
    
    if sys.argv[1] == '3':
        # Overlay Figure 3 cartoon
        overlayCartoon(fdir + 'figure3.svg',
                       f'{cartoon_dir}/figure3.svg', 65, 0, scalee=0.3)
    
    if sys.argv[1] == '4':
        # Overlay Figure 4 cartoon
        overlayCartoon(fdir + 'figure4.svg',
                       f'{cartoon_dir}/figure4.svg', 70, 0, scalee=0.3) 
        
    if sys.argv[1] == '5':
        # Overlay Figure 5 cartoon
        overlayCartoon(fdir + 'figure5.svg',
                       f'{cartoon_dir}/figure5.svg', 65, 0, scalee=0.3)
        
    if sys.argv[1] == '6':
        # Overlay Figure 6 cartoon
        overlayCartoon(fdir + 'figure6.svg',
                       f'{cartoon_dir}/figure6.svg', 10, 0, scalee=0.38)
    
    if sys.argv[1] == '7':
        # Overlay Figure 7 cartoon
        overlayCartoon(fdir + 'figure7.svg',
                       f'{cartoon_dir}/figure7.svg', 10, -10, scalee=0.38)
    
    if sys.argv[1] == '8':
        # Overlay Figure 8 cartoon
        overlayCartoon(fdir + 'figure8.svg',
                       f'{cartoon_dir}/figure8.svg', 40, 0, scalee=0.35)
    
    if sys.argv[1] == '9':
        # Overlay Figure 9 cartoon
        overlayCartoon(fdir + 'figure9.svg',
                       f'{cartoon_dir}/figure9.svg', 45, 0, scalee=0.32)
    
    if sys.argv[1] == '12':
        # Overlay Figure 12 cartoon
        overlayCartoon(fdir + 'figure12.svg',
                       f'{cartoon_dir}/figure2.svg', 65, 0, scalee=0.3) 
    
    if sys.argv[1] == '13':
        # Overlay Figure 13 cartoon
        overlayCartoon(fdir + 'figure13.svg',
                       f'{cartoon_dir}/figure3.svg', 65, 0, scalee=0.3)
        
    if sys.argv[1] == '14':
        # Overlay Figure 14 cartoon
        overlayCartoon(fdir + 'figure14.svg',
                       f'{cartoon_dir}/figure4.svg', 70, 0, scalee=0.3) 
    
    if sys.argv[1] == '15':
        # Overlay Figure 15 cartoon
        overlayCartoon(fdir + 'figure15.svg',
                       f'{cartoon_dir}/figure5.svg', 65, 0, scalee=0.3)
    
    if sys.argv[1] == '26':
        # Overlay Figure 26 cartoon
        overlayCartoon(fdir + 'figure26.svg',
                       f'{cartoon_dir}/figure3b.svg', 25, 30, scalee=0.23)
     
