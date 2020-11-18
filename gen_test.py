import numpy as np
import random


def gen_test(n_users, n_items):
    P = np.zeros((n_users, n_items))
    C = np.zeros((n_users, n_items))
    I = list(range(n_items))
    U = list(range(n_users))
    while len(I):
        group_items = []
        k = np.random.randint(5, 10)
        for _ in range(k):
            group_items.append(random.choice(I))
        I = [item for item in I if item not in group_items]
        k = np.random.randint(5, 10)
        group_users = random.sample(U, k)
        for i in group_items:
            for j in group_users:
                P[j, i] = 1
                C[j, i] = 1 + abs(np.random.normal(0, 0.1))
    np.savetxt('P.txt', P, delimiter=' ', fmt='%d')
    np.savetxt('C.txt', C, delimiter=' ', fmt='%.5f')


#gen_test(1000, 2000)


