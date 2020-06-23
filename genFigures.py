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
                       f'{cartoon_dir}/figure2.svg', 0, 0, scalee=0.2) 
    
    if sys.argv[1] == '3':
        # Overlay Figure 3 cartoon
        overlayCartoon(fdir + 'figure3.svg',
                       f'{cartoon_dir}/figure3.svg', 0, 0, scalee=0.18)
    
    if sys.argv[1] == '4':
        # Overlay Figure 4 cartoon
        overlayCartoon(fdir + 'figure4.svg',
                       f'{cartoon_dir}/figure4.svg', 0, 0, scalee=0.18) 
        
    if sys.argv[1] == '5':
        # Overlay Figure 5 cartoon
        overlayCartoon(fdir + 'figure5.svg',
                       f'{cartoon_dir}/figure5.svg', 0, 0, scalee=0.18)
        
    if sys.argv[1] == '6':
        # Overlay Figure 6 cartoon
        overlayCartoon(fdir + 'figure6.svg',
                       f'{cartoon_dir}/figure6.svg', 0, 0, scalee=0.19)
    
    if sys.argv[1] == '7':
        # Overlay Figure 7 cartoon
        overlayCartoon(fdir + 'figure7.svg',
                       f'{cartoon_dir}/figure7.svg', 0, 0, scalee=0.18)
    
    if sys.argv[1] == '8':
        # Overlay Figure 8 cartoon
        overlayCartoon(fdir + 'figure8.svg',
                       f'{cartoon_dir}/figure8.svg', 0, 0, scalee=0.18)
    
    if sys.argv[1] == '9':
        # Overlay Figure 9 cartoon
        overlayCartoon(fdir + 'figure9.svg',
                       f'{cartoon_dir}/figure9.svg', 0, 0, scalee=0.18)
    
  
