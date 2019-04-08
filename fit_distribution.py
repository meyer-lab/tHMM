import numpy as np
import pandas as pd
from lineage.fitting_distribution import check_dist

# read data into DataFrame

url1 = 'https://github.com/meyer-lab/lineage-growth/tree/master/lineage/G1_G2_duration_control.xlsx'
df = pd.read_excel(url1)

##----------------------- Preprocessing the data ------------------------##

# dataFrmae into numpy array
a = df.values
G1 = a[:, 0]
G2 = a[:, 1]

# removing nan from the array
G2 = G2[~np.isnan(G2)]

# converting from unit of [frames] into [hours]
# every frame is every 30 minutes, so dividing the numbers by 2 gives unit of [hours]
G1 = G1/2
G2 = G2/2


## --------------------- Check for our data ------------------------ ##
print('#### For G1 ####\n')
p_value = check_dist(G1, verbose = True)
print('\n #### For G2 ####\n')
p_value = check_dist(G2, verbose = True)

# What we get is:

#### probable distributions for G1: ####

# betaprime :    p-value =  0.9496245807703753
# gamma :    p-value =  0.7730477413285115
# erlang :    p-value =  0.7730478543522439

#### probable distributions for G2: ####

# betaprime :    p-value =  0.06029922688363665
# gamma :    p-value =  0.04329344124461376
# erlang :    p-value =  0.043293992146724136