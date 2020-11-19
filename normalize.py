import numpy as np


def sum_normalize(vector):
    total_sum = np.sum(vector)
    for i in range(len(vector)):
        vector[i] = float(vector[i] / total_sum)
    return vector


def softmax_normalize(vector):
    total_sum_exp = np.sum(np.exp(vector))
    for i in range(len(vector)):
        vector[i] = np.exp(vector[i]) / total_sum_exp
    return vector
