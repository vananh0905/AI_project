from WMF import WeightedMF
from IFConverter import IFConverter
import numpy as np

converter = IFConverter(alpha=10)
# converter.get_implicit_feedback('./data/TEST.csv')
# Loading fake data
with open('./data/R-train.txt', 'r') as f:
    converter.R = [[float(num) for num in line[:-1].split(' ')] for line in f]
    converter.R = np.array(converter.R)
converter.convert()
P, C = converter.P, converter.C
dict_item, dict_user = converter.get_dictionary()

wmf = WeightedMF(P, C, dict_user, dict_item, optimizer='sgd', depth=5, early_stopping=True, verbose=True)
if wmf.optimizer == 'formula':
    wmf.fit_formula()
else:
    wmf.fit_derivative()
print(wmf.get_recommendations(5, 20))
# Compare to original data
with open('./data/R-full.txt', 'r') as f:
    origin = [[float(num) for num in line[:-1].split(' ')] for line in f]
    origin = np.array(origin)
recommendations = np.argsort(origin[5])
recommendations = recommendations[-20:]
print(recommendations)
