import numpy as np


def ark(predict, target, k):
    if len(predict) > k:
        predict = predict[:k]
    scores = []
    for _ in range(1, k+1):
        true_pos = 0
        false_neg = 0
        for i in target:
            if i in predict[:_]:
                true_pos += 1
            if i not in predict[:_]:
                false_neg += 1
        scores.append(float(true_pos / (true_pos + false_neg)))
    return np.mean(scores)


def mark(predicts, targets, k):
    size_eval = len(targets)
    if len(predicts) > size_eval:
        predicts = predicts[:size_eval]
    scores = []
    for i in range(size_eval):
        scores.append(ark(predicts[i], targets[i], k))
    return np.mean(scores)