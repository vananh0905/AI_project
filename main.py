import numpy as np
from WMF import WeightedMF
from WMF import IFConverter

converter = IFConverter('./TEST.csv')
P, C = converter.get_data()
np.savetxt('P.txt', P, delimiter=' ', fmt='%d')
np.savetxt('C.txt', C, delimiter=' ', fmt='%5.3f')
with open('P.txt', 'r') as f:
    P = [[float(num) for num in line[:-1].split(' ')] for line in f]
    P = np.array(P)
with open('C.txt', 'r') as f:
    C = [[float(num) for num in line[:-1].split(' ')] for line in f]
    C = np.array(C)

# wmf = WeightedMF(P, C, depth=5, verbose=True)
# wmf.fit()
# # wmf.predict()
