import numpy as np
import random


def gen_test(n_users, n_items):
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
    np.savetxt('./data/R.txt', R, delimiter=' ', fmt='%d')


gen_test(100, 200)
