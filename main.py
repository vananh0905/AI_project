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

wmf = WeightedMF(P, C, dict_user, dict_item, optimizer='sgd', depth=10, early_stopping=True, verbose=True)
# wmf.fit()
# wmf.save()
wmf.load()

import evaluate
evaluate.eval_mark(original_path='./data/R-full.txt', model=wmf, k=20)
