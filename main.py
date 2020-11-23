from WMF import WeightedMF
from WMF import IFConverter

converter = IFConverter('./data/TEST.csv')
# Loading fake data. TEST.csv and this R.txt are irrelevant.
#converter.get_implicit_feedback()
#converter.save_R_real()
#converter.load_R_train()
#converter.convert()
#converter.save()
converter.load()
P, C = converter.P, converter.C
dict_item, dict_user = converter.get_dictionary()

wmf = WeightedMF(P, C, dict_user, dict_item, depth=5, early_stopping=True, verbose=True)
wmf.fit()
wmf.save()
