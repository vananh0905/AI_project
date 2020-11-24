import numpy as np
import random


def gen_fake_R(n_users, n_items):
    R = np.zeros((n_users, n_items))
    I = list(range(n_items))
    U = list(range(n_users))
    while len(I):
        group_items = []
        k = random.randint(5, 10)
        for _ in range(k):
            group_items.append(random.choice(I))
        I = [item for item in I if item not in group_items]
        k = random.randint(5, 10)
        group_users = random.sample(U, k)
        for j in group_items:
            for i in group_users:
                R[i, j] = int(abs(np.random.normal(loc=0, scale=0.1)) * 40) + 1
    np.savetxt('./data/R-full.txt', R, delimiter=' ', fmt='%d')


def gen_fake_R_train(path_to_R_full, gap_ratio=0.1):
    with open(path_to_R_full, 'r') as f:
        R = [[float(num) for num in line[:-1].split(' ')] for line in f]
        R = np.array(R)
        n_users = R.shape[0]
        n_items = R.shape[1]
        for i in range(n_users):
            for j in range(n_items):
                if R[i, j] > 0:
                    decision = random.random()
                    if decision < gap_ratio:
                        R[i, j] = 0
    np.savetxt('./data/R-train.txt', R, delimiter=' ', fmt='%d')


# gen_fake_R(100, 200)
gen_fake_R_train('./data/R-full.txt')
