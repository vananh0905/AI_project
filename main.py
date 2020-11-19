from WMF import WeightedMF
from WMF import IFConverter

converter = IFConverter('./data/TEST.csv')
# Loading fake data. TEST.csv and this R.txt are irrelevant.
converter.load()
converter.convert()
P, C = converter.P, converter.C

wmf = WeightedMF(P, C, depth=5, early_stopping=True, verbose=True)
wmf.fit()
wmf.save()
