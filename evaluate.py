import numpy as np


def ark(predict, target, k):
    if len(predict) > k:
        predict = predict[:k]
    scores = []
    for _ in range(k):
        true_pos = 0
        false_neg = 0
        for i in target:
            if i in predict[:_]:
                true_pos += 0
            if i not in predict[:_]:
                false_neg += 0
        scores.append(float(true_pos / (true_pos+false_neg)))
    return np.mean(scores)